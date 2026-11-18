import test from "node:test";
import assert from "node:assert/strict";
import { IDBFactory } from "fake-indexeddb";
import { NativeApiError } from "@agent-wechat/shared";
import { BrowserCache } from "../src/browser-cache.ts";
import { HubSession } from "../src/session.ts";

const page = (items, extra = {}) => ({
  items,
  nextCursor: null,
  coverage: {},
  sourceGeneration: "1",
  asOfEventCursor: "20",
  ...extra,
});
const message = (id, createdAt = "1", extra = {}) => ({
  id,
  serverId: id,
  accountId: "a",
  threadId: "peer",
  senderId: "peer",
  direction: "incoming",
  type: 1,
  body: id,
  createdAt,
  version: "1",
  media: [],
  ...extra,
});
const thread = (id = "peer", accountId = "a") => ({
  id,
  accountId,
  peerId: id,
  kind: "direct",
  unreadCount: 0,
  readStateVersion: "1",
  history: {},
});
function cache(t) {
  const values = new Map();
  const result = new BrowserCache("https://hub.test", "hub", {
    factory: new IDBFactory(),
    broadcast: false,
    storage: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value),
      removeItem: (key) => values.delete(key),
    },
  });
  t.after(() => result.close());
  return result;
}
async function seed(cache, cursor = "10", ownUsername) {
  await cache.writeAccount("a", {
    cursor,
    ownUsername,
    threads: [thread()],
    contacts: [],
    histories: [
      [
        "peer",
        {
          messages: [
            message("original"),
            message("quote", "2", {
              quote: {
                serverId: "original",
                state: "available",
                content: "private old excerpt",
                sender: "peer",
              },
            }),
          ],
          nextMessages: "old-page",
          sourceGeneration: "1",
          group: null,
        },
      ],
    ],
  });
  cache.rememberAccount("a");
  cache.saveView("a", { thread: "peer" });
}

test("sign-out hides history and retains the account-scoped cache and recovery state", async (t) => {
  const disk = cache(t);
  await seed(disk);
  disk.writeState("a", "peer", { text: "unfinished reply" }, "drafts");
  disk.writeState("a", "send:pending", { key: "pending" });
  const hub = session(t, disk, {
    events: async () => page([], { asOfEventCursor: "10" }),
  });
  await hub.activate("a");
  await hub.consume(
    {
      cursor: "21",
      kind: "account.updated",
      data: { authentication: "logged_out" },
    },
    "a",
  );
  await disk.flush();
  assert.equal(disk.lastAccount(), undefined);
  assert.equal((await disk.readAccount("a")).signedOut, true);
  assert.equal((await disk.readAccount("a")).cursor, "21");
  assert.deepEqual(hub.snapshot().messages, []);
  assert.deepEqual(disk.readState("a", "peer", "drafts"), {
    text: "unfinished reply",
  });
  assert.deepEqual(disk.readState("a", "send:pending"), { key: "pending" });
});

test("a missed logout during cache catch-up prevents history restoration", async (t) => {
  const disk = cache(t);
  await seed(disk);
  let historyReads = 0,
    streams = 0;
  const hub = session(t, disk, {
    events: async () =>
      page([
        {
          cursor: "20",
          kind: "account.updated",
          data: { authentication: "logged_out" },
        },
      ]),
    messages: async () => {
      historyReads++;
      return page([]);
    },
    async *watch() {
      streams++;
    },
  });
  await hub.activate("a");
  assert.equal(hub.snapshot().signInRequired, "a");
  assert.deepEqual(hub.snapshot().messages, []);
  assert.equal(historyReads, 0);
  assert.equal(streams, 0);
  await disk.flush();
  assert.equal((await disk.readAccount("a")).signedOut, true);
});

for (const startup of ["reopening", "signing back in"]) {
  test(`${startup} shows local history while the conversation list is still loading`, { timeout: 3000 }, async (t) => {
    const persistent = cache(t);
    await seed(persistent, "20", "self");
    if (startup === "signing back in") {
      await persistent.signedOut("a", "20");
      persistent.rememberAccount(null);
    }

    const inventory = Promise.withResolvers();
    const shown = Promise.withResolvers();
    const listed = Promise.withResolvers();
    const updated = Promise.withResolvers();
    const canonical = [
      message("original", "1", { body: null, recall: { actor: "self" } }),
      message("quote", "2", { quote: { serverId: "original", state: "recalled" } }),
    ];
    const newMessage = message("offline-delta", "3");
    const arrived = {
      cursor: "21",
      kind: "message.committed",
      data: { threadId: "peer", messageId: newMessage.id },
    };
    // Reopen receives a live event after the list snapshot. Re-login must also
    // catch a message between the early history and later inventory snapshots.
    const newInventoryBoundary = startup === "signing back in";
    const requested = [];
    const seen = [];
    let inventoryResolved = false;
    const hub = session(t, persistent, {
      accounts: async () => page([{ id: "a", authenticated: true, ownUsername: "self" }]),
      threads: async () => {
        await inventory.promise;
        inventoryResolved = true;
        return page([thread("other"), thread()], { asOfEventCursor: newInventoryBoundary ? "21" : "20" });
      },
      messages: async (account, id, { limit }) => {
        requested.push({ account, id, limit });
        return inventoryResolved
          ? page([...canonical, newMessage], { asOfEventCursor: "21" })
          : page(canonical);
      },
      events: async (_account, after) => page(inventoryResolved && after === "20" ? [arrived] : []),
      message: async () => newMessage,
      async *watch(_account, after, signal) {
        await inventory.promise;
        if (after === "20") yield arrived;
        await new Promise((resolve) =>
          signal.aborted ? resolve() : signal.addEventListener("abort", resolve, { once: true }),
        );
      },
    });
    const unsubscribe = hub.subscribe(() => {
      const { messages, threads } = hub.snapshot();
      seen.push(...messages);
      if (messages.some((item) => item.id === "quote")) shown.resolve();
      if (threads.some((item) => item.id === "other")) listed.resolve();
      if (messages.some((item) => item.id === newMessage.id)) updated.resolve();
    });
    t.after(unsubscribe);
    t.after(() => inventory.resolve());

    const activation = hub.activate("a");
    await shown.promise;
    assert.equal(inventoryResolved, false);
    assert.equal(hub.snapshot().selected, "peer");
    assert.equal(hub.snapshot().loading, false);
    assert.deepEqual(requested[0], { account: "a", id: "peer", limit: 40 });
    assert.equal(seen.some((item) => item.body === "original" || item.quote?.content === "private old excerpt"), false);

    inventory.resolve();
    await activation;
    await listed.promise;
    await updated.promise;
    assert.deepEqual(hub.snapshot().threads.map((item) => item.id), ["other", "peer"]);
    assert.equal(hub.snapshot().selected, "peer");
    assert.deepEqual(hub.snapshot().messages.map((item) => item.id), ["original", "quote", newMessage.id]);
  });
}

test("a delayed startup history response cannot restore a signed-out account", { timeout: 3000 }, async (t) => {
  const persistent = cache(t);
  await seed(persistent, "20", "self");
  const inventory = Promise.withResolvers();
  const started = Promise.withResolvers();
  const history = Promise.withResolvers();
  let authenticated = true;
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated, ownUsername: "self" }]),
    threads: async () => inventory.promise,
    messages: async () => {
      started.resolve();
      return history.promise;
    },
  });
  t.after(() => {
    history.resolve(page([]));
    inventory.resolve(page([]));
  });

  const activation = hub.activate("a");
  await started.promise;
  authenticated = false;
  await hub.accounts();
  history.resolve(page([message("late-private-content")]));
  inventory.resolve(page([thread()]));
  await activation;
  assert.equal(hub.snapshot().signInRequired, "a");
  assert.deepEqual(hub.snapshot().messages, []);
  assert.equal(hub.snapshot().selected, null);
});

test("an unseen sign-out before the local history boundary prevents warm restoration", async (t) => {
  const persistent = cache(t);
  await seed(persistent, "19", "self");
  const seen = [];
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated: true, ownUsername: "self" }]),
    events: async () => page([{
      cursor: "20",
      kind: "account.updated",
      data: { authentication: "logged_out" },
    }]),
    messages: async () => page([message("private-history")]),
  });
  t.after(hub.subscribe(() => seen.push(...hub.snapshot().messages)));

  await hub.activate("a");
  assert.equal(hub.snapshot().signInRequired, "a");
  assert.deepEqual(seen, []);
  assert.deepEqual(hub.snapshot().messages, []);
});

test("finishing the startup inventory preserves the chat the user opened in the meantime", { timeout: 3000 }, async (t) => {
  const persistent = cache(t);
  await persistent.writeAccount("a", {
    cursor: "20",
    ownUsername: "self",
    threads: [thread(), thread("other")],
    contacts: [],
    histories: [],
  });
  persistent.saveView("a", { thread: "peer" });
  const inventory = Promise.withResolvers();
  const shown = Promise.withResolvers();
  const listed = Promise.withResolvers();
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated: true, ownUsername: "self" }]),
    threads: async () => inventory.promise,
    messages: async (_account, id) => page([message(id, "1", { threadId: id })]),
  });
  t.after(() => inventory.resolve(page([])));
  t.after(hub.subscribe(() => {
    const { messages, threads } = hub.snapshot();
    if (messages.some((item) => item.id === "peer")) shown.resolve();
    if (threads[0]?.displayName === "Refreshed") listed.resolve();
  }));

  const activation = hub.activate("a");
  await shown.promise;
  await hub.select("other");
  inventory.resolve(page([{ ...thread(), displayName: "Refreshed" }, thread("other")]));
  await activation;
  await listed.promise;
  assert.equal(hub.snapshot().selected, "other");
  assert.deepEqual(hub.snapshot().messages.map((item) => item.id), ["other"]);
});

test("switching chats during startup replay cannot restore a recalled body or quoted excerpt", { timeout: 3000 }, async (t) => {
  const persistent = cache(t);
  const original = message("original");
  const quote = message("quote", "2", {
    quote: { serverId: original.serverId, state: "available", content: "old excerpt" },
  });
  const recalled = [
    { ...original, body: null, recall: { actor: "self" } },
    { ...quote, quote: { serverId: original.serverId, state: "recalled" } },
  ];
  await persistent.writeAccount("a", {
    cursor: "20",
    ownUsername: "self",
    threads: [thread(), thread("other")],
    contacts: [],
    histories: [],
  });
  persistent.saveView("a", { thread: "peer" });
  const replayPaused = Promise.withResolvers();
  const finishReplay = Promise.withResolvers();
  const finishHistory = Promise.withResolvers();
  let peerReads = 0;
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated: true, ownUsername: "self" }]),
    threads: async () => page([thread(), thread("other")], { asOfEventCursor: "22" }),
    events: async (_account, after) => {
      if (after === "20") return page([{
        cursor: "21",
        kind: "message.recalled",
        data: { messageId: original.id, threadId: original.threadId },
      }]);
      replayPaused.resolve();
      await finishReplay.promise;
      return page([{ cursor: "22", kind: "push.hint", data: {} }]);
    },
    messages: async (_account, id) => {
      if (id === "other") return page([message("other", "1", { threadId: id })]);
      if (++peerReads === 1) return page([original, quote]);
      await finishHistory.promise;
      return page(recalled, { asOfEventCursor: "22" });
    },
  });
  t.after(() => {
    finishReplay.resolve();
    finishHistory.resolve();
  });
  const stale = () => hub.snapshot().messages.some((item) =>
    item.body === original.body || item.quote?.content === quote.quote.content,
  );

  const activation = hub.activate("a");
  await replayPaused.promise;
  assert.equal(stale(), false);
  await hub.select("other");
  const reopen = hub.select("peer");
  assert.equal(hub.snapshot().selected, "peer");
  assert.equal(stale(), false);
  finishHistory.resolve();
  await reopen;
  finishReplay.resolve();
  await activation;
  assert.equal(hub.snapshot().messages.find((item) => item.id === quote.id).quote.state, "recalled");
  assert.equal(stale(), false);
});

function session(t, cache, overrides = {}, now = Date.now) {
  const hub = new HubSession(
    {
      accounts: async () =>
        page([
          { id: "a", authenticated: true },
          { id: "b", authenticated: true },
        ]),
      threads: async (account) =>
        page([thread("other", account), thread("peer", account)]),
      thread: async (account, id) => thread(id, account),
      messages: async () =>
        page([message("latest", "100")], { nextCursor: "fresh-page" }),
      events: async () => ({
        items: [{ cursor: "20", kind: "push.hint", data: {} }],
        nextCursor: "20",
      }),
      contacts: async () => page([]),
      capabilities: async () => ({}),
      async *watch(_account, _after, signal) {
        await new Promise((resolve) =>
          signal.aborted
            ? resolve()
            : signal.addEventListener("abort", resolve, { once: true }),
        );
      },
      ...overrides,
    },
    undefined,
    cache,
    now,
  );
  t.after(() => hub.close());
  return hub;
}

test("reload restores last chat, catches up a missed recall, and redacts older quoted excerpts", async (t) => {
  const persistent = cache(t);
  await seed(persistent);
  persistent.saveBlob(
    "media",
    "a",
    "preview",
    { blob: new Blob(["old photo"]), checkedAt: 1 },
    "original",
  );
  const hub = session(t, persistent, {
    events: async () => ({
      items: [
        {
          cursor: "20",
          kind: "message.recalled",
          data: { messageId: "original" },
        },
      ],
    }),
    message: async () =>
      message("original", "1", { body: null, recall: { actor: "self" } }),
    messages: async () => page([
      message("original", "1", { body: null, recall: { actor: "self" } }),
      message("quote", "2", { quote: { serverId: "original", state: "recalled" } }),
    ]),
  });
  hub.onMediaInvalidated((account, id) =>
    persistent.invalidateMessage(account, id),
  );
  const seen = [];
  hub.subscribe(() => seen.push(...hub.snapshot().messages));
  await hub.activate("a");
  assert.equal(hub.snapshot().selected, "peer");
  assert.equal(
    hub.snapshot().messages.find((item) => item.id === "original").body,
    null,
  );
  assert.equal(
    hub.snapshot().messages.find((item) => item.id === "quote").quote.state,
    "recalled",
  );
  assert.equal(
    seen.some(
      (item) =>
        item.body === "original" ||
        item.quote?.content === "private old excerpt",
    ),
    false,
  );
  await persistent.flush();
  assert.equal(await persistent.blob("media", "a", "preview"), undefined);
  assert.equal((await persistent.readAccount("a")).cursor, "20");
});

for (const reason of ["expired events", "rewound journal", "changed source"])
  test(`${reason} discards older persisted history before rendering`, async (t) => {
    const persistent = cache(t);
    await seed(persistent, reason === "rewound journal" ? "30" : "10");
    persistent.saveBlob(
      "media",
      "a",
      "verified-image",
      {
        blob: new Blob(["verified image bytes"]),
        checkedAt: 1,
      },
      "original",
    );
    const hub = session(t, persistent, {
      ...(reason === "expired events"
        ? {
            events: async () => {
              throw new NativeApiError(410, "event_cursor_expired");
            },
          }
        : {}),
      messages: async () =>
        page([message("latest", "100")], {
          nextCursor: "fresh-page",
          sourceGeneration: reason === "changed source" ? "2" : "1",
        }),
    });
    await hub.activate("a");
    assert.deepEqual(
      hub.snapshot().messages.map((item) => item.id),
      ["latest"],
    );
    await persistent.flush();
    assert.equal((await persistent.readAccount("a")).cursor, "20");
    assert.equal(
      await (await persistent.blob("media", "a", "verified-image")).blob.text(),
      "verified image bytes",
    );
  });

test("an unauthenticated account cannot read its persistent cache", async (t) => {
  const persistent = cache(t);
  await seed(persistent);
  let reads = 0;
  const read = persistent.readAccount.bind(persistent);
  persistent.readAccount = async (account) => {
    reads++;
    return read(account);
  };
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated: false }]),
  });
  await assert.rejects(hub.activate("a"), { code: "auth_required" });
  assert.equal(reads, 0);
  assert.equal(persistent.lastAccount(), undefined);
  assert.ok(await read("a"));
  assert.deepEqual(hub.snapshot().messages, []);
});

test("a failed first page cannot make unvalidated persisted bodies visible on retry", async (t) => {
  const persistent = cache(t);
  await seed(persistent);
  const hub = session(t, persistent, {
    messages: async () => {
      throw Error("offline");
    },
  });
  await hub.activate("a");
  assert.deepEqual(hub.snapshot().messages, []);
  let finish;
  hub.client.messages = () =>
    new Promise((resolve) => {
      finish = resolve;
    });
  const retry = hub.select("peer");
  for (let i = 0; i < 20 && !finish; i++)
    await new Promise((resolve) => setTimeout(resolve, 1));
  assert.deepEqual(hub.snapshot().messages, []);
  finish(page([message("latest", "100")], { sourceGeneration: "2" }));
  await retry;
  assert.deepEqual(
    hub.snapshot().messages.map((item) => item.id),
    ["latest"],
  );
});

test("late account validation cannot switch away from the more recent activation", async (t) => {
  const persistent = cache(t);
  let finish;
  const late = new Promise((resolve) => {
    finish = resolve;
  });
  let calls = 0;
  const hub = session(t, persistent, {
    accounts: async () =>
      ++calls === 1 ? late : page([{ id: "b", authenticated: true }]),
  });
  const first = hub.activate("a");
  await hub.activate("b");
  finish(page([{ id: "a", authenticated: true }]));
  await first;
  assert.equal(hub.snapshot().account, "b");
  assert.equal(persistent.lastAccount(), "b");
});

for (const scenario of ["unchanged", "expired TTL", "changed last ID", "missing last ID", "missing timestamp", "future timestamp"]) {
  test(`restored history: ${scenario}`, async (t) => {
    const persistent = cache(t);
    const now = 1000000;
    await persistent.writeAccount("a", {
      cursor: "20", threads: [thread()], contacts: [],
      histories: [["peer", { messages: [message("tail")], nextMessages: null, sourceGeneration: "1", group: null,
        ...(scenario === "missing timestamp" ? {} : { validatedAt: scenario === "expired TTL" ? now - 300000 : scenario === "future timestamp" ? now + 1 : now - 1000 }),
      }]],
    });
    persistent.saveView("a", { thread: "peer" });
    let reads = 0;
    const current = { ...thread(), ...(scenario === "missing last ID" ? {} : { lastMessage: { id: scenario === "changed last ID" ? "new" : "tail" } }) };
    const hub = session(t, persistent, {
      threads: async () => page([current]), thread: async () => current,
      messages: async () => { reads++; return page([message("authoritative")]); },
    }, () => now);
    await hub.activate("a");
    assert.equal(reads, scenario === "unchanged" ? 0 : 1);
    assert.equal(hub.snapshot().messages[0].id, scenario === "unchanged" ? "tail" : "authoritative");
  });
}

test("logging back into the same account refreshes its history once and skips its old sign-out event", async (t) => {
  const persistent = cache(t);
  let reads = 0, authenticated = true;
  const current = { ...thread(), lastMessage: { id: "tail" } };
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated, ownUsername: "self" }]),
    threads: async () => page([current]), thread: async () => current,
    messages: async () => { reads++; return page([message("tail")]); },
  });
  await hub.activate("a");
  const initialReads = reads;
  authenticated = false;
  await hub.accounts();
  assert.deepEqual(hub.snapshot().messages, []);
  await persistent.flush();
  assert.equal((await persistent.readAccount("a")).signedOut, true);
  authenticated = true;
  // A poll-detected logout has no exact event cursor; its earlier event can
  // still be in the replay after a fresh authenticated login.
  hub.client.events = async () => ({ items: [{ cursor: "21", kind: "account.updated", data: { authentication: "logged_out" } }] });
  hub.client.threads = async () => page([current], { asOfEventCursor: "21" });
  await hub.accounts();
  await hub.activate("a");
  assert.equal(reads - initialReads, 1);
  assert.equal(hub.snapshot().messages[0].id, "tail");
});

test("a different authenticated profile identity cannot restore the previous account cache", async (t) => {
  const persistent = cache(t);
  await persistent.writeAccount("a", { cursor: "20", ownUsername: "old-owner", contacts: [], threads: [thread()],
    histories: [["peer", { messages: [message("tail")], validatedAt: 1000000, nextMessages: null, sourceGeneration: "1", group: null }]],
  });
  const current = { ...thread(), lastMessage: { id: "tail" } };
  let reads = 0;
  const seen = [];
  const hub = session(t, persistent, {
    accounts: async () => page([{ id: "a", authenticated: true, ownUsername: "new-owner" }]),
    threads: async () => page([current]), thread: async () => current,
    messages: async () => { reads++; return page([message("new-owner-content")]); },
  }, () => 1000000);
  hub.subscribe(() => seen.push(...hub.snapshot().messages));
  await hub.activate("a");
  assert.equal(reads, 1);
  assert.equal(seen.some((item) => item.id === "tail"), false);
});

test("cache catch-up stops at the inventory boundary so newer updates stay on the live stream", async (t) => {
  const persistent = cache(t);
  await seed(persistent, "10");
  let resumedAfter;
  const hub = session(t, persistent, {
    events: async () => ({ items: [
      { cursor: "20", kind: "push.hint", data: {} },
      { cursor: "21", kind: "message.committed", data: { threadId: "peer", messageId: "new" } },
    ] }),
    async *watch(_account, after, signal) {
      resumedAfter = after;
      await new Promise((done) => signal.addEventListener("abort", done, { once: true }));
    },
  });
  await hub.activate("a");
  assert.equal(resumedAfter, "20");
});
