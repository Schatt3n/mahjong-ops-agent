import {
  NativeApiError,
  NativeWeChatClient,
  isNativeTombstone,
  type NativeContact,
  type NativeGroup,
  type NativeEvent,
  type NativeMessage,
  type NativePage,
  type NativeSendIntent,
  type NativeThread,
  type QuoteSendReference,
  type NativeThreadMessage,
  type TextMentions,
} from "@agent-wechat/shared";
import { allMentionProblem, canMentionAll, mentionOptions, type MentionToken } from "./mentions.ts";
import type { BrowserCache } from "./browser-cache.ts";
import { HISTORY_TTL_MS, HistoryCache } from "./history-cache.ts";
import { t } from "./i18n.ts";
import { messageSummary } from "./thread-preview.ts";
import { canReply } from "./reply.ts";
import { quotedText, quoteTargetMatches } from "./quote-preview.ts";

export type Account = {
  id: string;
  alias?: string;
  ownUsername?: string | null;
  authenticated?: boolean;
  connection?: string;
};
export type PendingSend = {
  key: string;
  text: string;
  threadId: string;
  quote?: QuoteSendReference;
  quotePreview?: NativeThreadMessage;
  mentions?: TextMentions;
  intent?: NativeSendIntent;
  uncertain?: boolean;
  dismissed?: boolean;
  error?: string;
};
export type SessionState = {
  account: string | null;
  accounts: Account[];
  threads: NativeThread[];
  contacts: NativeContact[];
  contactPreview: NativeContact[] | null;
  contactsLoading: boolean;
  messages: NativeMessage[];
  sends: PendingSend[];
  selected: string | null;
  selectedGroup: NativeGroup | null;
  nextMessages: string | null;
  loading: boolean;
  opening: boolean;
  connected: boolean;
  reconnecting: boolean;
  signInRequired: string | null;
  error: string | null;
  capabilities: Record<string, unknown>;
};
const initial = (): SessionState => ({
  account: null,
  accounts: [],
  threads: [],
  contacts: [],
  contactPreview: null,
  contactsLoading: false,
  messages: [],
  sends: [],
  selected: null,
  selectedGroup: null,
  nextMessages: null,
  loading: false,
  opening: false,
  connected: false,
  reconnecting: false,
  signInRequired: null,
  error: null,
  capabilities: {},
});
function isSignOut(event: NativeEvent): boolean {
  return (
    [
      "account.updated",
      "account.connection",
      "auth.updated",
      "account.push",
    ].includes(event.kind) &&
    (event.data.authentication === "logged_out" ||
      event.data.authenticated === false ||
      event.data.state === "auth_required" ||
      event.data.reason === "auth_required")
  );
}
export function readableError(error: unknown): string {
  if (error instanceof NativeApiError)
    return (
      (
        {
          unauthorized: t("session.agentWechatHubAccessKey"),
          forbidden: t("session.accessKeyDoesnTAllow"),
          auth_required: t("session.signWechatAgain"),
          operation_in_progress: t("session.profileHasOperationProgressWait"),
          account_sign_out_required: t("session.signOutBeforeRemovingProfile"),
          account_confirmation_mismatch: t("session.profileConfirmationDoesNotMatch"),
          read_state_conflict: t("session.conversationChangedRefreshTryAgain"),
          idempotency_conflict: t("session.sendKeyAlreadyUse"),
          invalid_mentions: t("session.chooseMentionedMembersAgain"),
          invalid_mention_text: t("session.messageMentionPositionsInvalidEdit"),
          mentions_require_group: t("session.mentionsAvailableGroupsOnly"),
          mention_member_unavailable: t("session.mentionedMemberUnavailableRemoveMention"),
          mention_group_membership_unavailable: t("session.refreshGroupSMembersBefore"),
          mention_all_forbidden: t("session.canTMentionEveryoneGroup"),
          mention_all_permission_unavailable: t("session.groupPermissionsUnavailableRefreshGroup"),
          quote_mentions_unsupported: t("session.hubCannotCombineMentionsQuoted"),
          quote_target_unavailable: t("session.messageNoLongerAvailableQuote"),
          quote_target_mismatch: t("session.quotedMessageChangedSelectAgain"),
          quote_target_ambiguous: t("session.messageCannotIdentifiedReliablyChoose"),
          quote_content_unavailable: t("session.contentMessageCannotQuoted"),
          quote_sender_unavailable: t("session.senderMessageUnavailable"),
          unsupported_quote_type: t("session.messageTypeCannotQuotedYet"),
          invalid_quote_reference: t("session.selectMessageWantReplyAgain"),
          invalid_quote_request: t("session.selectMessageWantReplyAgain"),
          unsupported_simulator_operation: t("session.actionIsnTAvailableSimulator"),
          unsupported_application_media: t("session.messageDoesnTContainSupported"),
          unsupported_group_creation: t("session.groupCreationIsnTAvailable"),
          invalid_group_members: t("session.chooseDistinctContactsGroupCreation"),
          ordinary_contact_required: t("session.oneTheseContactsCannotAdded"),
          ordinary_group_required: t("session.refreshGroupCheckStillMember"),
          group_state_incomplete: t("session.refreshGroupBeforeChangingSettings"),
          operation_cannot_cancel: t("session.changeHasStartedCanNo"),
          invalid_group_text: t("session.useNameMost512Utf"),
          pagination_cursor_expired: t("session.historyViewExpiredReopenConversation"),
        } as Record<string, string>
      )[error.code] ??
      t("session.agentWechatHubRequestFailed", { code: error.code })
    );
  return t("session.couldnTReachAgentWechat");
}
function timestamp(value: string | null): number {
  return value && /^\d+$/.test(value)
    ? Number(value)
    : Date.parse(value ?? "") || 0;
}
const compareMessages = (a: NativeMessage, b: NativeMessage): number =>
  timestamp(a.createdAt) - timestamp(b.createdAt) ||
  a.id.localeCompare(b.id, undefined, { numeric: true });
export function mergeMessage(
  messages: NativeMessage[],
  message: NativeMessage,
): NativeMessage[] {
  const recalled = new Set(
    messages
      .filter(
        (item) =>
          item.recall &&
          item.accountId === message.accountId &&
          item.threadId === message.threadId,
      )
      .map((item) => item.serverId)
      .filter(Boolean),
  );
  if (message.recall)
    recalled.add(
      message.serverId ??
        messages.find((item) => item.id === message.id)?.serverId ??
        null,
    );
  return [
    ...messages.filter(
      (item) =>
        item.id !== message.id &&
        !(
          message.serverId &&
          item.threadId === message.threadId &&
          item.serverId === message.serverId
        ),
    ),
    message,
  ]
    .map((item) =>
      item.accountId === message.accountId &&
      item.threadId === message.threadId &&
      item.quote &&
      recalled.has(item.quote.serverId)
        ? {
            ...item,
            quote: {
              serverId: item.quote.serverId,
              type: item.quote.type,
              state: "recalled",
            },
          }
        : item,
    )
    .sort(compareMessages);
}
export function pause(
  milliseconds: number,
  signal: AbortSignal,
): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const done = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", done);
      resolve();
    };
    const timer = setTimeout(done, milliseconds);
    signal.addEventListener("abort", done, { once: true });
  });
}
export async function inventory<T>(
  read: (cursor?: string) => Promise<NativePage<T>>,
  firstPage?: (items: T[]) => void,
): Promise<{ items: T[]; cursor: string }> {
  const items = new Map<string, T>();
  let next: string | undefined,
    boundary = "0";
  const seen = new Set<string>();
  for (let pageNumber = 0; pageNumber < 500; pageNumber++) {
    const page = await read(next);
    if (!next) boundary = page.asOfEventCursor ?? "0";
    for (const item of page.items) {
      if (isNativeTombstone(item)) items.delete(item.id);
      else items.set((item as { id: string }).id, item);
    }
    if (pageNumber === 0 && firstPage) firstPage([...items.values()]);
    if (!page.nextCursor)
      return { items: [...items.values()], cursor: boundary };
    if (seen.has(page.nextCursor)) throw new Error("Repeated inventory cursor");
    seen.add(page.nextCursor);
    next = page.nextCursor;
  }
  throw new Error("Inventory exceeds the supported view size");
}

/** Each account activation and thread selection fences every late response. */
export class HubSession {
  private value = initial();
  private listeners = new Set<() => void>();
  private signOutListeners = new Set<(account: string) => void>();
  onSignedOut = (listener: (account: string) => void): (() => void) => {
    this.signOutListeners.add(listener);
    return () => {
      this.signOutListeners.delete(listener);
    };
  };
  private mediaInvalidationListeners = new Set<
    (account: string, messageId: string) => void
  >();
  onMediaInvalidated = (
    listener: (account: string, messageId: string) => void,
  ): (() => void) => {
    this.mediaInvalidationListeners.add(listener);
    return () => {
      this.mediaInvalidationListeners.delete(listener);
    };
  };
  private invalidateMedia(account: string, messageId: string): void {
    for (const listener of this.mediaInvalidationListeners)
      listener(account, messageId);
  }
  private settingsListeners = new Set<(account: string) => void>();
  onAccountSettings = (listener: (account: string) => void): (() => void) => {
    this.settingsListeners.add(listener);
    return () => {
      this.settingsListeners.delete(listener);
    };
  };
  private mediaListeners = new Set<(assetId: string) => void>();
  onMediaReady = (listener: (assetId: string) => void): (() => void) => {
    this.mediaListeners.add(listener);
    return () => {
      this.mediaListeners.delete(listener);
    };
  };
  private generation = 0;
  private threadGeneration = 0;
  private groupRequest = 0;
  private accountRequest = 0;
  private accountsCheckedAt = -Infinity;
  private catalogRequest = 0;
  private conversationRefresh?: {
    account: string;
    generation: number;
    running: boolean;
    again: boolean;
    catalog: boolean;
    groups: Set<string>;
    promise: Promise<void>;
  };
  private pendingMessageRefresh?: {
    cursor: string;
    flight?: Promise<void>;
    error?: string;
  };
  private detailsRequest = 0;
  private threadRequests = new Map<string, number>();
  private interactionReads = new Set<string>();
  private catalogFlight: {
    account: string;
    generation: number;
    promise: Promise<void>;
  } | null = null;
  private stream?: AbortController;
  private cursor = "0";
  private pending = new Map<string, PendingSend[]>();
  private histories = new HistoryCache();
  private historyLoaded = false;
  private historySource = "";
  private historyValidatedAt?: number;
  private stopped = false;
  private cache?: BrowserCache;
  private restoring = false;
  private activationRequest = 0;
  private failedInitialInventory?: { preferred?: string };
  private reconcileFlight?: Promise<void>;
  private restored = new Set<string>();
  private replayReady = false;
  private readonly now: () => number;
  private historyRequest?: AbortController;
  readonly client: NativeWeChatClient;
  private receipts?: {
    read(account: string): PendingSend[];
    write(account: string, sends: PendingSend[]): void;
  };
  constructor(
    client: NativeWeChatClient,
    receipts?: {
      read(account: string): PendingSend[];
      write(account: string, sends: PendingSend[]): void;
    },
    cache?: BrowserCache,
    now: () => number = Date.now,
    initialAccounts?: Account[],
  ) {
    this.client = client;
    this.receipts = receipts;
    this.cache = cache;
    this.now = now;
    if (initialAccounts) {
      this.value.accounts = initialAccounts;
      this.accountsCheckedAt = this.now();
    }
  }
  snapshot = (): SessionState => this.value;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  private current(account: string, generation: number): boolean {
    return (
      !this.stopped &&
      generation === this.generation &&
      account === this.value.account
    );
  }
  private publish(patch: Partial<SessionState>): void {
    if (this.stopped) return;
    this.value = { ...this.value, ...patch };
    if (patch.sends && this.value.account) {
      this.pending.set(this.value.account, patch.sends);
      this.receipts?.write(this.value.account, patch.sends);
    }
    for (const listener of this.listeners) listener();
  }
  clearError(): void {
    this.publish({ error: null });
  }
  private persist(): void {
    const account = this.value.account;
    if (!account || !this.cache || this.restoring || this.failedInitialInventory || this.stopped)
      return;
    void this.cache.writeAccount(account, {
      threads: this.value.threads,
      contacts: this.value.contacts,
      histories: this.histories.export(account),
      cursor: this.pendingMessageRefresh?.cursor ?? this.cursor,
      ownUsername:
        this.value.accounts.find((item) => item.id === account)?.ownUsername ??
        undefined,
    });
  }
  async clearLocalCache(account = this.value.account): Promise<void> {
    if (!account) return;
    this.histories.clear(account);
    this.restored.clear();
    await this.cache?.clearAccount(account);
  }
  invalidateLocalCache(account: string): void {
    this.histories.clear(account);
    this.restored.clear();
    if (this.value.account === account) {
      this.historyLoaded = false;
      this.historySource = "";
      this.historyValidatedAt = undefined;
    }
  }
  private remember(): void {
    const { account, selected, messages, nextMessages, selectedGroup } =
      this.value;
    if (this.historyLoaded && account && selected)
      this.histories.set(account, selected, {
        messages,
        nextMessages,
        group: selectedGroup,
        sourceGeneration: this.historySource,
        validatedAt: this.historyValidatedAt,
      });
    this.persist();
  }
  async accounts(): Promise<Account[]> {
    const request = ++this.accountRequest,
      generation = this.generation;
    const result = await this.client.accounts();
    const accounts = result.items.filter(
      (item) => typeof item.id === "string",
    ) as Account[];
    if (
      !this.stopped &&
      request === this.accountRequest &&
      generation === this.generation
    ) {
      this.accountsCheckedAt = this.now();
      this.acceptAccounts(accounts);
    }
    return accounts;
  }
  private acceptAccounts(accounts: Account[]): void {
    const account = this.value.account;
    if (
      account &&
      accounts.some(
        (item) => item.id === account && item.authenticated === false,
      )
    ) {
      this.value = { ...this.value, accounts };
      this.signedOut(account);
    } else this.publish({ accounts });
  }
  private signedOut(account: string, cursor?: string): void {
    if (this.value.account !== account) return;
    this.resetAccount(account, true, cursor);
    for (const listener of this.signOutListeners) listener(account);
  }
  close(): void {
    this.activationRequest++;
    this.generation++;
    this.threadGeneration++;
    this.stream?.abort();
    this.historyRequest?.abort();
    this.publish(initial());
    this.stopped = true;
    this.mediaListeners.clear();
    this.settingsListeners.clear();
    this.mediaInvalidationListeners.clear();
    this.signOutListeners.clear();
    this.histories.clear();
  }
  deactivate(account = this.value.account): void {
    this.resetAccount(account, false);
  }
  private resetAccount(
    account: string | null,
    signedOut: boolean,
    cursor?: string,
  ): void {
    this.activationRequest++;
    if (account) {
      if (this.value.account === account) this.remember();
      this.histories.clear(account);
      if (this.cache?.lastAccount() === account)
        this.cache.rememberAccount(null);
      void this.cache?.signedOut(account, cursor);
    }
    if (this.value.account !== account) return;
    this.generation++;
    this.threadGeneration++;
    this.catalogRequest++;
    this.stream?.abort();
    this.historyRequest?.abort();
    this.historyLoaded = false;
    this.historySource = "";
    this.historyValidatedAt = undefined;
    this.restoring = false;
    this.replayReady = false;
    this.restored.clear();
    this.pendingMessageRefresh = undefined;
    this.failedInitialInventory = undefined;
    this.cursor = "0";
    this.publish({
      ...initial(),
      accounts: this.value.accounts.map((item) =>
        signedOut && item.id === account
          ? { ...item, authenticated: false, connection: "disconnected" }
          : item,
      ),
      signInRequired: signedOut ? account : null,
    });
  }
  /** 用会话列表的边界校验本地缓存，再从该游标启动事件流。 */
  private async readInitialInventory(
    account: string,
    generation: number,
    signal: AbortSignal,
  ): Promise<string | undefined> {
    const delays = [500, 1500];
    for (let attempt = 0; ; attempt++) {
      if (signal.aborted || !this.current(account, generation)) return;
      try {
        return await this.refresh(account, generation, false);
      } catch (error) {
        if (error instanceof NativeApiError && error.code === "auth_required") {
          if (this.current(account, generation)) this.signedOut(account);
          return;
        }
        const delay = delays[attempt];
        if (
          delay === undefined ||
          (error instanceof NativeApiError && !error.retryable && error.status < 500)
        ) throw error;
        await pause(delay, signal);
      }
    }
  }
  /** 回放到指定快照的事件边界，作废过时的正文和媒体缓存。 */
  private async replayStartupEvents(
    account: string,
    generation: number,
    boundary: string,
    previousSignOut: boolean,
  ): Promise<string | null> {
    const through = BigInt(boundary);
    if (BigInt(this.cursor) > through) throw new Error("Source journal rewound");
    let changedSelection: string | null = null;
    for (let page = 0; BigInt(this.cursor) < through; page++) {
      if (page >= 10) throw new Error("Catch-up exceeds cache limit");
      const before = this.cursor;
      const events = await this.client.events(account, before, 200);
      for (const event of events.items) {
        if (!this.current(account, generation)) return null;
        if (BigInt(event.cursor) > through) break;
        if (["message.committed", "message.recalled"].includes(event.kind)) {
          const histories = this.histories.export(account);
          const messageId = event.data.messageId;
          const matching = histories.find(([, history]) =>
            history.messages.some((message) => message.id === messageId),
          );
          if (event.kind === "message.recalled" && typeof messageId === "string") {
            this.invalidateMedia(account, messageId);
            const original = matching?.[1].messages.find((message) => message.id === messageId);
            if (original?.serverId)
              for (const [, history] of histories)
                for (const message of history.messages)
                  if (message.quote?.serverId === original.serverId)
                    this.invalidateMedia(account, message.id);
          }
          const affected =
            typeof event.data.threadId === "string" ? event.data.threadId : matching?.[0];
          if (affected) this.histories.delete(account, affected);
          else this.histories.clear(account);
          const selected = this.value.selected;
          if (selected && (!affected || affected === selected)) {
            changedSelection = selected;
            // 切换聊天时 remember 不应把已经失效的正文重新放回内存缓存。
            this.historyLoaded = false;
            this.historySource = "";
            if (event.kind === "message.recalled") {
              // 旧快照的迟到响应也不能把已撤回的消息补回界面。
              this.historyRequest?.abort();
              const serverId = typeof event.data.serverId === "string"
                ? event.data.serverId
                : matching?.[1].messages.find((message) => message.id === messageId)?.serverId;
              const messages = typeof messageId !== "string" && typeof serverId !== "string"
                ? []
                : this.value.messages
                    .filter((message) => message.id !== messageId && (!serverId || message.serverId !== serverId))
                    .map((message) => serverId && message.quote?.serverId === serverId
                      ? { ...message, quote: { serverId, type: message.quote.type, state: "recalled" as const } }
                      : message);
              this.publish({ messages, nextMessages: null });
            }
          }
        } else if (event.kind === "send.updated" && typeof event.data.id === "string") {
          // 恢复发送回执时不触发全量列表查询，启动流程稍后会统一读取。
          await this.acceptReceipt(
            await this.client.send(account, event.data.id),
            account,
            generation,
            undefined,
            false,
          );
        } else if (
          !(previousSignOut && isSignOut(event)) &&
          ![
            "sync.completed",
            "thread.read",
            "thread.read-state",
            "group.updated",
            "thread.updated",
            "contacts.updated",
            "metadata.changed",
          ].includes(event.kind)
        ) {
          await this.consume(event, account, generation);
        }
        if (!this.current(account, generation)) return null;
        this.cursor = event.cursor;
      }
      if (!this.current(account, generation)) return null;
      if (this.cursor === before) throw new Error("Incomplete cache catch-up");
    }
    return changedSelection;
  }

  /** 优先读取上次的聊天。只显示本机 Hub 当前快照中的正文，先核对到该快照为止的退出事件。 */
  private async restoreLastConversation(
    account: string,
    thread: string,
    generation: number,
    previousSignOut: boolean,
    ownUsername: string,
  ): Promise<boolean> {
    const selected = this.threadGeneration;
    const request = (this.historyRequest = new AbortController());
    const current = () =>
      this.current(account, generation) && selected === this.threadGeneration;
    try {
      const validatedAt = this.now();
      const page = await this.client.messages(account, thread, { limit: 40 }, request.signal);
      if (!current() || page.asOfEventCursor == null) return false;
      await this.replayStartupEvents(account, generation, page.asOfEventCursor, previousSignOut);
      // 已知的退出事件可能早于本次扫码，只有当前登录仍有效才恢复会话。
      if (previousSignOut && current()) {
        await this.verifyRestoredAccount(account, generation, ownUsername);
      }
      if (!current()) return false;
      const messages = page.items
        .filter((item): item is NativeMessage => !isNativeTombstone(item))
        .reduce(mergeMessage, []);
      this.restored.delete(JSON.stringify([account, thread]));
      this.historyLoaded = true;
      this.historySource = page.sourceGeneration;
      this.historyValidatedAt = validatedAt;
      this.publish({
        selected: thread,
        messages,
        nextMessages: page.nextCursor,
        loading: false,
        opening: true,
      });
      this.cache?.saveView(account, { thread });
      this.remember();
      return true;
    } catch (error) {
      if (current() && error instanceof NativeApiError && error.code === "auth_required")
        this.signedOut(account);
      return false;
    }
  }

  /** 重新扫码时区分旧的退出事件和当前账号状态。 */
  private async verifyRestoredAccount(
    account: string,
    generation: number,
    ownUsername?: string,
  ): Promise<void> {
    const accounts = await this.accounts();
    if (this.current(account, generation) && !accounts.some((item) =>
      item.id === account && item.authenticated === true &&
      (!ownUsername || item.ownUsername === ownUsername),
    )) {
      this.signedOut(account);
    }
  }

  /** 登录后先恢复上次的本地会话，再刷新列表并从它的事件边界接续实时更新。 */
  async activate(account: string, preferred?: string): Promise<void> {
    const activation = ++this.activationRequest;
    if (this.cache) {
      const accounts =
        this.now() - this.accountsCheckedAt < 5000 &&
        this.now() >= this.accountsCheckedAt
          ? this.value.accounts
          : await this.accounts();
      if (activation !== this.activationRequest || this.stopped) return;
      if (
        !accounts.some(
          (item) => item.id === account && item.authenticated === true,
        )
      ) {
        this.cache.rememberAccount(null);
        throw new NativeApiError(409, "auth_required");
      }
    }
    this.remember();
    this.historyLoaded = false;
    this.replayReady = false;
    this.historyValidatedAt = undefined;
    this.stream?.abort();
    this.historyRequest?.abort();
    const generation = ++this.generation;
    const startupSelection = ++this.threadGeneration;
    this.catalogRequest++;
    this.pendingMessageRefresh = undefined;
    this.failedInitialInventory = undefined;
    this.cursor = "0";
    const sends =
      this.pending.get(account) ?? this.receipts?.read(account) ?? [];
    this.publish({
      account,
      threads: [],
      contacts: [],
      contactPreview: null,
      contactsLoading: false,
      messages: [],
      sends,
      selected: null,
      selectedGroup: null,
      nextMessages: null,
      loading: true,
      opening: false,
      connected: false,
      reconnecting: false,
      signInRequired: null,
      error: null,
      capabilities: {},
    });
    const controller = (this.stream = new AbortController());
    this.restoring = !!this.cache;
    this.cache?.rememberAccount(account);
    try {
      let saved = this.cache
        ? await this.cache.readAccount(account)
        : undefined;
      if (!this.current(account, generation)) return;
      const own = this.value.accounts.find(
        (item) => item.id === account,
      )?.ownUsername;
      if (saved && own && saved.ownUsername !== own) {
        await this.cache?.clearAccount(account);
        if (!this.current(account, generation)) return;
        this.histories.clear(account);
        this.restored.clear();
        saved = undefined;
      }
      if (saved) {
        this.cursor = saved.cursor;
        this.publish({ threads: saved.threads, contacts: saved.contacts });
        for (const [thread, history] of saved.histories) {
          this.histories.set(account, thread, history);
          this.restored.add(JSON.stringify([account, thread]));
        }
      }
      const candidate = preferred ?? this.cache?.view(account).thread;
      const warmed = !!(
        own && saved?.ownUsername === own && candidate &&
        saved.threads.some((thread) => thread.id === candidate) &&
        startupSelection === this.threadGeneration &&
        await this.restoreLastConversation(account, candidate, generation, !!saved.signedOut, own)
      );
      if (!this.current(account, generation)) return;
      const warmBoundary = this.cursor;
      let boundary: string | undefined;
      try {
        boundary = await this.readInitialInventory(account, generation, controller.signal);
      } catch (error) {
        if (this.current(account, generation)) {
          this.failedInitialInventory = { preferred };
          this.restoring = false;
          this.publish({ error: readableError(error), loading: false, opening: false });
        }
        return;
      }
      if (boundary === undefined || !this.current(account, generation)) return;
      let changedSelection: string | null = null;
      if (saved) {
        try {
          changedSelection = await this.replayStartupEvents(
            account, generation, boundary, !!saved.signedOut,
          );
        } catch (error) {
          if (!this.current(account, generation)) return;
          if (error instanceof NativeApiError && error.code === "auth_required") {
            this.signedOut(account);
            return;
          }
          changedSelection = this.value.selected;
          this.historyLoaded = false;
          this.historySource = "";
          this.publish({ messages: [], nextMessages: null });
          this.histories.clear(account);
          this.restored.clear();
          await this.cache?.clearHistory(account);
          if (!this.current(account, generation)) return;
          this.cursor = boundary;
        }
      } else this.cursor = boundary;
      if (!this.current(account, generation)) return;
      if (saved?.signedOut && (!warmed || boundary !== warmBoundary)) {
        await this.verifyRestoredAccount(account, generation, own ?? undefined);
        if (!this.current(account, generation)) return;
      }
      this.replayReady = true;
      this.restoring = false;
      if (changedSelection && changedSelection === this.value.selected) {
        // 保留仍有效的正文，补读期间继续显示；失效的缓存不能参与旧页合并。
        await this.select(changedSelection, false, true);
      } else if (startupSelection === this.threadGeneration) {
        const listed = (id?: string) => this.value.threads.find((thread) => thread.id === id);
        const target = listed(preferred) ?? listed(this.cache?.view(account).thread) ?? this.value.threads[0];
        if (!target) {
          this.historyLoaded = false;
          this.publish({ selected: null, messages: [], nextMessages: null, loading: false, opening: false });
        } else if (warmed && target.id === this.value.selected && target.unreadCount === 0) {
          this.publish({ opening: false });
          if (target.id.endsWith("@chatroom"))
            void this.refreshGroup(account, target.id, generation, startupSelection).catch(() => {});
        } else {
          await this.select(target.id);
        }
      }
    } catch (error) {
      if (this.current(account, generation))
        this.publish({ error: readableError(error), loading: false, opening: false });
    }
    if (this.current(account, generation)) {
      this.restoring = false;
      this.persist();
      void this.watch(account, generation, controller.signal);
      void this.loadCatalog(account, generation).catch(() => {});
      for (const send of sends) void this.checkSend(send.key);
    }
  }
  async reconcile(): Promise<void> {
    if (this.reconcileFlight) return this.reconcileFlight;
    const account = this.value.account,
      generation = this.generation;
    if (!account || this.value.opening || this.value.loading) return;
    this.reconcileFlight = (async () => {
      if (this.failedInitialInventory) {
        await this.refresh(account, generation);
        return;
      }
      if (this.value.connected) {
        const selected = this.value.selected;
        if (selected && !this.fresh(this.histories.get(account, selected)))
          await this.select(selected, true);
        return;
      }
      await this.accounts();
      if (!this.current(account, generation)) return;
      const boundary = await this.refresh(account, generation, false);
      if (!this.current(account, generation)) return;
      if (BigInt(this.cursor) > BigInt(boundary))
        await this.recover(account, generation);
      else if (this.value.selected)
        await this.select(this.value.selected, true);
    })()
      .catch(() => {})
      .finally(() => {
        this.reconcileFlight = undefined;
      });
    return this.reconcileFlight;
  }
  private fresh(history: { validatedAt?: number } | undefined): boolean {
    const at = history?.validatedAt;
    return (
      at !== undefined &&
      Number.isFinite(at) &&
      this.now() >= at &&
      this.now() - at < HISTORY_TTL_MS
    );
  }
  /** 打开会话，先加载可显示的消息，再按已取得的会话版本确认未读状态。 */
  async select(threadId: string, force = false, keepVisible = false): Promise<void> {
    const account = this.value.account;
    if (!account) return;
    this.remember();
    const cached = this.histories.get(account, threadId);
    this.historyRequest?.abort();
    const request = (this.historyRequest = new AbortController());
    const restoredKey = JSON.stringify([account, threadId]);
    const fromDisk = this.restored.has(restoredKey);
    const mergeable = !fromDisk || this.replayReady ? cached : undefined;
    const visible = keepVisible && this.value.selected === threadId ? this.value.messages : [];
    this.cache?.saveView(account, { thread: threadId });
    this.historyLoaded = !!cached && !fromDisk;
    this.historySource = mergeable?.sourceGeneration ?? "";
    this.historyValidatedAt = mergeable?.validatedAt;
    const previousGroup =
      this.value.selected === threadId
        ? this.value.selectedGroup
        : cached?.group ?? null;
    const generation = this.generation,
      selected = ++this.threadGeneration;
    const current = () =>
      !request.signal.aborted && this.current(account, generation) && selected === this.threadGeneration;
    this.publish({
      selected: threadId,
      selectedGroup: previousGroup ? { ...previousGroup, canMentionAll: null } : null,
      messages: fromDisk ? [] : cached?.messages ?? visible,
      nextMessages: mergeable?.nextMessages ?? null,
      loading: visible.length === 0,
      opening: true,
      error: null,
    });
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        const baseline =
          attempt === 0 ? mergeable?.messages ?? visible : this.value.messages;
        const listed = this.value.threads.find(
          (thread) => thread.id === threadId,
        );
        const canReuseHistory =
          attempt === 0 &&
          cached &&
          !force &&
          (fromDisk ? this.replayReady : this.value.connected) &&
          this.fresh(cached);
        // 会话列表已提供读取边界。用它直接加载消息，避免先串行读取会话详情。
        // 如果版本过期，后端会拒绝标记已读，再重新读取边界和历史。
        // 复用缓存正文时还需要最新消息的摘要，列表缺少摘要则读取详情。
        const readBoundary =
          attempt === 0 &&
          !force &&
          listed &&
          (canReuseHistory
            ? listed.lastMessage !== undefined
            : listed.readStateVersion)
            ? listed
            : await this.refreshThread(
                account,
                threadId,
                generation,
                request.signal,
              );
        if (!current()) return;
        if (
          canReuseHistory &&
          readBoundary.lastMessage !== undefined &&
          (readBoundary.lastMessage?.id ?? null) ===
            (cached.messages.at(-1)?.id ?? null) &&
          readBoundary.unreadCount === 0
        ) {
          this.historyLoaded = true;
          this.restored.delete(restoredKey);
          this.publish({ messages: cached.messages, loading: false });
          return; // Replay maintains this cache only until its history TTL expires.
        }
        const validatedAt = this.now();
        const page = await this.client.messages(
          account,
          threadId,
          {
            limit: 40,
          },
          request.signal,
        );
        if (!current()) return;
        const latest = (
          page.items.filter(
            (item) => !isNativeTombstone(item),
          ) as NativeMessage[]
        ).reduce(mergeMessage, []);
        // Preserve older loaded pages, replace the current server range, and apply
        // only events received after this request on top of its snapshot.
        const older =
          page.nextCursor &&
          latest.length &&
          this.historySource === page.sourceGeneration
            ? baseline.filter((item) => compareMessages(item, latest[0]) < 0)
            : [];
        const arrivals = this.value.messages.filter(
          (item) => !baseline.includes(item),
        );
        const removed = new Set(
          page.items.filter(isNativeTombstone).map((item) => item.id),
        );
        const messages = arrivals
          .reduce(mergeMessage, latest.reduce(mergeMessage, older))
          .filter((item) => !removed.has(item.id));
        const sameSource = this.historySource === page.sourceGeneration;
        this.restored.delete(restoredKey);
        this.historyLoaded = true;
        this.historySource = page.sourceGeneration;
        this.historyValidatedAt = validatedAt;
        this.publish({
          messages,

          nextMessages:
            page.nextCursor && sameSource
              ? this.value.nextMessages
              : page.nextCursor,
          loading: false,
        });
        this.remember();
        if ((readBoundary.unreadCount ?? 0) <= 0 || !readBoundary.readStateVersion)
          return;
        try {
          await this.client.markRead(
            account,
            threadId,
            readBoundary.readStateVersion,
          );
          if (this.current(account, generation))
            await this.refreshThread(account, threadId, generation);
          return;
        } catch (error) {
          if (!current()) return;
          // A conflict admits no read. Reload both version and history once;
          // never clear unseen messages by retrying only with a newer version.
          if (
            error instanceof NativeApiError &&
            error.code === "read_state_conflict" &&
            attempt === 0
          )
            continue;
          this.publish({
            error: t("session.couldnTClearUnreadStatus", { name: readableError(error) }),
          });
          return;
        }
      }
    } catch (error) {
      if (current())
        this.publish({ error: readableError(error), loading: false });
    } finally {
      if (current()) {
        this.publish({ opening: false });
        if (threadId.endsWith("@chatroom"))
          void this.refreshGroup(account, threadId, generation, selected).catch(
            () => {},
          );
      }
    }
  }
  async older(): Promise<void> {
    const { account, selected, nextMessages } = this.value,
      generation = this.generation,
      threadGeneration = this.threadGeneration,
      signal = this.historyRequest?.signal;
    if (!account || !selected || !nextMessages || this.value.loading || signal?.aborted) return;
    this.publish({ loading: true });
    try {
      const page = await this.client.messages(
        account,
        selected,
        {
          cursor: nextMessages,
          limit: 100,
        },
        signal,
      );
      if (
        signal?.aborted ||
        !this.current(account, generation) ||
        threadGeneration !== this.threadGeneration
      )
        return;
      const messages = this.value.messages.reduce(
        mergeMessage,
        (
          page.items.filter(
            (item) => !isNativeTombstone(item),
          ) as NativeMessage[]
        ).reduce(mergeMessage, []),
      );
      this.publish({ messages, nextMessages: page.nextCursor, loading: false });
      this.historySource = page.sourceGeneration;
      this.remember();
    } catch (error) {
      if (
        !signal?.aborted &&
        this.current(account, generation) &&
        threadGeneration === this.threadGeneration
      )
        if (
          error instanceof NativeApiError &&
          error.code === "pagination_cursor_expired"
        ) {
          this.histories.clear(account);
          await this.select(selected);
        } else this.publish({ loading: false, error: readableError(error) });
    }
  }
  async revealMessage(id: string, signal: AbortSignal): Promise<boolean> {
    const { account, selected } = this.value;
    const generation = this.generation,
      threadGeneration = this.threadGeneration;
    const current = () =>
      !signal.aborted &&
      !!account &&
      this.current(account, generation) &&
      threadGeneration === this.threadGeneration;
    while (current()) {
      if (
        this.value.messages.some(
          (message) =>
            message.id === id &&
            message.accountId === account &&
            message.threadId === selected,
        )
      )
        return true;
      if (this.value.loading) {
        await pause(50, signal);
        continue;
      }
      const cursor = this.value.nextMessages;
      if (!cursor) return false;
      await this.older();
      if (this.value.nextMessages === cursor) return false;
    }
    return false;
  }
  async revealQuote(
    parent: NativeMessage,
    signal: AbortSignal,
  ): Promise<NativeMessage | undefined> {
    const { account, selected } = this.value;
    const generation = this.generation,
      threadGeneration = this.threadGeneration;
    if (
      !parent.quote ||
      parent.accountId !== account ||
      parent.threadId !== selected
    )
      return;
    const current = () =>
      !signal.aborted &&
      !!account &&
      this.current(account, generation) &&
      threadGeneration === this.threadGeneration;
    while (current()) {
      const matches = this.value.messages.filter((message) =>
        quoteTargetMatches(parent, message),
      );
      if (matches.length) return matches.length === 1 ? matches[0] : undefined;
      if (this.value.loading) {
        await pause(50, signal);
        continue;
      }
      const cursor = this.value.nextMessages;
      if (!cursor) return;
      await this.older();
      if (this.value.nextMessages === cursor) return;
    }
  }
  private async refreshThread(
    account: string,
    threadId: string,
    generation: number,
    signal?: AbortSignal,
  ): Promise<NativeThread> {
    const key = JSON.stringify([account, threadId]);
    const inventoryVersion = this.catalogRequest;
    const request = (this.threadRequests.get(key) ?? 0) + 1;
    this.threadRequests.set(key, request);
    let thread: NativeThread;
    try {
      thread = await this.client.thread(account, threadId, signal);
    } catch (error) {
      if (!(error instanceof NativeApiError) || error.status !== 404)
        throw error;
      // A new direct conversation has no session row until its first message.
      return {
        id: threadId,
        accountId: account,
        peerId: threadId,
        kind: threadId.endsWith("@chatroom") ? "group" : "direct",
        displayName: null,
        unreadCount: 0,
        readStateVersion: "",
        history: {},
      };
    }
    if (
      this.current(account, generation) &&
      inventoryVersion === this.catalogRequest &&
      this.threadRequests.get(key) === request
    ) {
      const found = this.value.threads.some((item) => item.id === threadId);
      this.publish({
        threads: found
          ? this.value.threads.map((item) =>
              item.id === threadId ? { ...item, ...thread } : item,
            )
          : [thread, ...this.value.threads],
      });
    }
    return thread;
  }
  private async refreshGroup(
    account: string,
    thread: string,
    generation: number,
    selected: number,
  ): Promise<void> {
    const request = this.invalidateAllMentionPermission();
    const group = await this.client.group(account, thread);
    if (
      this.current(account, generation) &&
      selected === this.threadGeneration &&
      thread === this.value.selected &&
      request === this.groupRequest &&
      group.id === thread
    ) {
      this.publish({ selectedGroup: group });
      this.remember();
    }
  }
  private invalidateAllMentionPermission(): number {
    const request = ++this.groupRequest;
    const group = this.value.selectedGroup;
    if (group && group.canMentionAll !== null)
      this.publish({ selectedGroup: { ...group, canMentionAll: null } });
    return request;
  }
  async markRead(threadId: string): Promise<unknown> {
    const account = this.value.account,
      generation = this.generation;
    if (!account) throw new Error("Select a conversation first.");
    const thread = await this.refreshThread(account, threadId, generation);
    const result = await this.client.markRead(
      account,
      threadId,
      thread.readStateVersion,
    );
    if (this.current(account, generation))
      await this.refreshThread(account, threadId, generation);
    return result;
  }
  async readOnInteraction(account: string, threadId: string): Promise<void> {
    if (
      this.stopped ||
      account !== this.value.account ||
      threadId !== this.value.selected ||
      !this.historyLoaded ||
      this.restoring ||
      this.failedInitialInventory ||
      this.value.opening
    )
      return;
    const thread = this.value.threads.find((item) => item.id === threadId);
    if (!thread?.readStateVersion || (thread.unreadCount ?? 0) <= 0) return;
    const generation = this.generation;
    const version = thread.readStateVersion;
    const key = JSON.stringify([account, threadId, version]);
    if (this.interactionReads.has(key)) return;
    this.interactionReads.add(key);
    try {
      // Focusing or replying acknowledges this view, never a newer unseen version.
      await this.client.markRead(account, threadId, version);
    } catch (error) {
      if (
        this.current(account, generation) &&
        this.value.selected === threadId &&
        !this.value.error &&
        !(
          error instanceof NativeApiError &&
          error.code === "read_state_conflict"
        )
      )
        this.publish({
          error: t("session.couldnTClearUnreadStatus", { name: readableError(error) }),
        });
    } finally {
      if (this.current(account, generation))
        await this.refreshThread(account, threadId, generation).catch(() => {});
      this.interactionReads.delete(key);
    }
  }
  /** 首次没有缓存时先显示第一页，完整读取成功后才替换并保存通讯录。 */
  private loadCatalog(account: string, generation: number): Promise<void> {
    if (
      this.catalogFlight?.account === account &&
      this.catalogFlight.generation === generation
    )
      return this.catalogFlight.promise;
    const request = ++this.detailsRequest;
    const promise = (async () => {
      const [contacts, capabilities, accounts] = await Promise.all([
        inventory(
          (cursor) => this.client.contacts(account, { cursor, limit: 200 }),
          (first) => {
            if (this.current(account, generation) && request === this.detailsRequest && !this.value.contacts.length)
              this.publish({ contactPreview: first });
          },
        ),
        this.client.capabilities(account),
        this.now() - this.accountsCheckedAt < 5000
          ? Promise.resolve({ items: this.value.accounts })
          : this.client.accounts(),
      ]);
      if (
        this.current(account, generation) &&
        request === this.detailsRequest
      ) {
        this.acceptAccounts(
          accounts.items.filter(
            (item) => typeof item.id === "string",
          ) as Account[],
        );
        if (!this.current(account, generation)) return;
        this.publish({
          contacts: contacts.items,
          contactPreview: null,
          capabilities,
        });
        this.persist();
      }
    })().finally(() => {
      if (this.catalogFlight?.promise !== promise) return;
      this.catalogFlight = null;
      if (this.current(account, generation) && request === this.detailsRequest)
        this.publish({ contactsLoading: false, contactPreview: null });
    });
    this.catalogFlight = { account, generation, promise };
    this.publish({ contactsLoading: true });
    return promise;
  }
  async refresh(
    account = this.value.account,
    generation = this.generation,
    catalog = true,
  ): Promise<string> {
    if (!account) return "0";
    if (this.current(account, generation) && this.failedInitialInventory) {
      try {
        await this.activate(account, this.failedInitialInventory.preferred);
      } catch (error) {
        if (this.current(account, generation)) this.publish({ error: readableError(error) });
        throw error;
      }
      return this.current(account, generation + 1) ? this.cursor : "0";
    }
    if (catalog && this.current(account, generation) && this.value.selected?.endsWith("@chatroom"))
      this.invalidateAllMentionPermission();
    const request = ++this.catalogRequest;
    const revisions = new Map(this.threadRequests);
    const threads = await inventory((cursor) =>
      this.client.threads(account, { cursor, limit: 200 }),
    );
    if (this.current(account, generation) && request === this.catalogRequest) {
      // A point read begun during this inventory may carry a newer setting.
      // Restart at page one instead of splicing it into an older page order.
      if (
        threads.items.some(
          (thread) =>
            revisions.get(JSON.stringify([account, thread.id])) !==
            this.threadRequests.get(JSON.stringify([account, thread.id])),
        )
      )
        return this.refresh(account, generation, catalog);
      const refreshError = this.pendingMessageRefresh?.error;
      this.pendingMessageRefresh = undefined;
      this.publish({
        threads: threads.items,
        ...(refreshError && refreshError === this.value.error ? { error: null } : {}),
      });
      this.persist();
    }
    if (catalog && this.current(account, generation)) {
      await Promise.all([
        this.loadCatalog(account, generation),
        this.value.selected?.endsWith("@chatroom")
          ? this.refreshGroup(
              account,
              this.value.selected,
              generation,
              this.threadGeneration,
            ).catch(() => {})
          : null,
      ]);
    }
    return threads.cursor;
  }
  /** 合并消息、资料更新和写入结果引起的会话列表刷新。 */
  refreshConversations(
    account = this.value.account,
    catalog = false,
    group?: string,
  ): Promise<void> {
    const generation = this.generation;
    if (!account || !this.current(account, generation))
      return Promise.resolve();
    if (this.value.selected?.endsWith("@chatroom") && (catalog || group === this.value.selected))
      this.invalidateAllMentionPermission();
    // Invalidate older inventories and point reads immediately, before the timer.
    this.catalogRequest++;
    const previous = this.conversationRefresh;
    if (previous?.account === account && previous.generation === generation) {
      previous.again ||= previous.running;
      previous.catalog ||= catalog;
      if (group) previous.groups.add(group);
      return previous.promise;
    }
    const work = {
      account,
      generation,
      running: false,
      again: false,
      catalog,
      groups: new Set(group ? [group] : []),
      promise: Promise.resolve(),
    };
    this.conversationRefresh = work;
    work.promise = (async () => {
      await pause(50, this.stream?.signal ?? new AbortController().signal);
      do {
        if (!this.current(account, generation)) return;
        work.again = false;
        work.running = true;
        const withCatalog = work.catalog;
        work.catalog = false;
        const groups = [...work.groups];
        work.groups.clear();
        const before = this.catalogRequest;
        await this.refresh(account, generation, withCatalog);
        if (!this.current(account, generation)) return;
        if (this.catalogRequest !== before + 1) work.again = true;
        if (!withCatalog && groups.includes(this.value.selected ?? ""))
          await this.refreshGroup(
            account,
            this.value.selected!,
            generation,
            this.threadGeneration,
          ).catch(() => {});
      } while (work.again || work.catalog || work.groups.size);
      this.persist();
    })().finally(() => {
      if (this.conversationRefresh === work)
        this.conversationRefresh = undefined;
    });
    return work.promise;
  }
  /** 消息仍按事件顺序入缓存，列表读取在后台合并。列表落盘前保留旧游标以便重放。 */
  private refreshAfterStreamMessage(account: string, generation: number): void {
    const pending = (this.pendingMessageRefresh ??= { cursor: this.cursor });
    const flight = this.refreshConversations(account);
    if (pending.flight === flight) return;
    pending.flight = flight;
    const current = () =>
      this.current(account, generation) &&
      this.pendingMessageRefresh === pending &&
      pending.flight === flight;
    void flight.catch(async (error) => {
      if (!current()) return;
      if (error instanceof NativeApiError && error.code === "auth_required") {
        this.signedOut(account);
        return;
      }
      pending.error = readableError(error);
      this.publish({ error: pending.error });
      const signal = this.stream?.signal;
      if (!signal) return;
      // 列表读取单独重试，期间事件流可以继续处理消息。
      await pause(1500, signal);
      if (!signal.aborted && current())
        this.refreshAfterStreamMessage(account, generation);
    });
  }
  async refreshMessage(account: string, messageId: string): Promise<void> {
    if (this.value.account !== account) return;
    await this.canonical(account, messageId, this.generation);
  }
  private async canonical(
    account: string,
    messageId: string,
    generation: number,
  ): Promise<NativeMessage | undefined> {
    const message = await this.client.message(account, messageId);
    if (!this.current(account, generation)) return;
    if (message.recall) {
      this.invalidateMedia(account, messageId);
      const messages =
        message.threadId === this.value.selected
          ? this.value.messages
          : this.histories.get(account, message.threadId)?.messages ?? [];
      const server =
        message.serverId ??
        messages.find((item) => item.id === message.id)?.serverId;
      for (const quoted of messages) {
        if (
          server &&
          quoted.accountId === account &&
          quoted.threadId === message.threadId &&
          quoted.quote?.serverId === server
        )
          this.invalidateMedia(account, quoted.id);
      }
    }
    if (message.threadId === this.value.selected) {
      this.publish({ messages: mergeMessage(this.value.messages, message) });
      this.remember();
    } else {
      const cached = this.histories.get(account, message.threadId);
      if (cached)
        this.histories.set(account, message.threadId, {
          ...cached,
          messages: mergeMessage(cached.messages, message),
        });
    }
    this.publish({
      sends: this.value.sends.filter(
        (send) =>
          send.intent?.localMessageId !== message.id &&
          !(message.serverId && send.intent?.serverId === message.serverId),
      ),
    });
    this.updatePreview(message);
    return message;
  }
  private updatePreview(message: NativeMessage): void {
    const senderName =
      this.value.contacts.find((item) => item.id === message.senderId)
        ?.displayName ?? null;
    this.publish({
      threads: this.value.threads.map((thread) => {
        if (thread.id !== message.threadId) return thread;
        const last = thread.lastMessage;
        if (
          last &&
          last.id !== message.id &&
          (timestamp(last.createdAt) > timestamp(message.createdAt) ||
            (timestamp(last.createdAt) === timestamp(message.createdAt) &&
              last.id.localeCompare(message.id, undefined, { numeric: true }) >=
                0))
        )
          return thread;
        return { ...thread, lastMessage: messageSummary(message, senderName) };
      }),
    });
  }
  private async acceptReceipt(
    intent: NativeSendIntent,
    account: string,
    generation: number,
    key?: string,
    refreshList = true,
  ): Promise<void> {
    if (!this.current(account, generation) || intent.accountId !== account)
      return;
    let sends = this.value.sends;
    const settled = sends.find((send) => send.intent?.id === intent.id)?.intent;
    // Accepted and cancelled intents cannot return to the queue. An admission
    // response or status read may arrive after cancellation or an SSE update.
    if (
      settled &&
      ["accepted", "cancelled"].includes(settled.state) &&
      intent.state !== settled.state
    )
      intent = settled;
    if (key && sends.some((send) => send.key === key))
      sends = sends.filter(
        (send) => send.key === key || send.intent?.id !== intent.id,
      );
    const known = sends.some(
      (send) => send.key === key || send.intent?.id === intent.id,
    );
    if (!known && !key && !intent.localMessageId)
      sends = [
        ...sends,
        {
          key: `observed:${intent.id}`,
          text: t("session.messageAnotherClient"),
          threadId: intent.destination,
          intent,
        },
      ];
    this.publish({
      sends: sends.map((send) =>
        send.key === key || send.intent?.id === intent.id
          ? { ...send, intent, uncertain: false, error: undefined }
          : send,
      ),
    });
    if (intent.localMessageId) {
      await this.canonical(account, intent.localMessageId, generation);
      if (this.current(account, generation))
        this.publish({
          sends: this.value.sends.filter(
            (send) => send.key !== key && send.intent?.id !== intent.id,
          ),
        });
      if (refreshList) await this.refreshConversations(account);
    }
  }
  async consume(
    event: NativeEvent,
    account: string,
    generation = this.generation,
    fromStream = false,
  ): Promise<void> {
    if (!this.current(account, generation)) return;
    if (isSignOut(event)) {
      this.signedOut(account, event.cursor);
      return;
    }
    if (event.kind === "account.settings_updated")
      for (const listener of this.settingsListeners) listener(account);
    if (
      event.kind === "message.recalled" &&
      typeof event.data.messageId === "string"
    )
      this.invalidateMedia(account, event.data.messageId);
    if (
      event.kind === "media.ready" &&
      typeof event.data.assetId === "string"
    ) {
      for (const listener of this.mediaListeners) listener(event.data.assetId);
    }
    if (
      (event.kind === "message.committed" ||
        event.kind === "message.recalled") &&
      typeof event.data.messageId === "string"
    ) {
      const message = await this.canonical(
        account,
        event.data.messageId,
        generation,
      );
      if (message && this.current(account, generation)) {
        if (fromStream) this.refreshAfterStreamMessage(account, generation);
        else await this.refreshConversations(account);
      }
    } else if (
      event.kind === "send.updated" &&
      typeof event.data.id === "string"
    ) {
      await this.acceptReceipt(
        await this.client.send(account, event.data.id),
        account,
        generation,
      );
    } else if (["thread.read", "thread.read-state"].includes(event.kind)) {
      if (typeof event.data.threadId === "string")
        await this.refreshThread(account, event.data.threadId, generation);
      else await this.refresh(account, generation, false);
    } else if (["thread.updated", "group.updated"].includes(event.kind)) {
      await this.refreshConversations(
        account,
        false,
        typeof event.data.groupId === "string" ? event.data.groupId
          : event.kind === "group.updated" && typeof event.data.threadId === "string" ? event.data.threadId : undefined,
      );
    } else if (
      ["account.updated", "account.connection", "auth.updated"].includes(
        event.kind,
      )
    ) {
      await this.accounts();
    } else if (
      event.kind === "contacts.updated" &&
      event.data.origin === "contact_sync"
    ) {
      void this.refreshConversations(account, true).catch((error) => {
        if (this.current(account, generation))
          this.publish({ error: readableError(error) });
      });
    } else if (
      event.kind === "contacts.updated" ||
      (event.kind === "metadata.changed" && event.data.kind !== "avatar")
    ) {
      await this.refreshConversations(account, true);
    } else if (event.kind === "sync.completed") {
      await this.refreshConversations(account);
    }
    // Push hints and control diagnostics carry no conversation content.
  }
  private async recover(account: string, generation: number): Promise<void> {
    const boundary = await this.refresh(account, generation);
    if (!this.current(account, generation)) return;
    this.histories.clear(account);
    this.restored.clear();
    await this.cache?.clearHistory(account);
    if (!this.current(account, generation)) return;
    this.historyLoaded = false;
    const selected = this.value.selected;
    if (selected) await this.select(selected);
    if (this.current(account, generation)) this.cursor = boundary;
  }
  private async watch(
    account: string,
    generation: number,
    signal: AbortSignal,
  ): Promise<void> {
    while (!signal.aborted && this.current(account, generation)) {
      try {
        if (this.value.reconnecting) await this.refreshConversations(account);
        if (signal.aborted || !this.current(account, generation)) return;
        for await (const event of this.client.watch(
          account,
          this.cursor,
          signal,
          {
            onActivity: () => {
              if (
                this.current(account, generation) &&
                !signal.aborted &&
                !this.value.connected
              )
                this.publish({
                  connected: true,
                  reconnecting: false,
                  error: null,
                });
            },
          },
        )) {
          if (signal.aborted || !this.current(account, generation)) return;
          const metadataOnly =
            [
              "thread.updated",
              "group.updated",
              "contacts.updated",
              "sync.completed",
            ].includes(event.kind) ||
            (event.kind === "metadata.changed" && event.data.kind !== "avatar");
          const applied = this.consume(event, account, generation, true);
          if (metadataOnly) {
            // Do not serialize an entire contact-sync burst into one read per
            // event. The shared refresh persists the final order when complete.
            void applied.catch((error) => {
              if (this.current(account, generation))
                this.publish({ error: readableError(error) });
            });
          } else await applied;
          if (this.current(account, generation)) {
            this.cursor = event.cursor;
            if (metadataOnly) continue;
            if (
              [
                "message.committed",
                "message.recalled",
                "group.updated",
                "sync.completed",
                "contacts.updated",
                "metadata.changed",
                "thread.read",
                "thread.read-state",
              ].includes(event.kind)
            )
              this.persist();
            else if (!this.pendingMessageRefresh)
              void this.cache?.checkpoint(account, this.cursor);
          }
        }
        if (!signal.aborted && this.current(account, generation))
          this.publish({ connected: false, reconnecting: true });
      } catch (error) {
        if (signal.aborted || !this.current(account, generation)) return;
        if (error instanceof NativeApiError && error.code === "auth_required") {
          this.signedOut(account);
          return;
        }
        this.publish({
          connected: false,
          reconnecting: true,
          error: readableError(error),
        });
        if (
          error instanceof NativeApiError &&
          ["cursor_expired", "event_cursor_expired"].includes(error.code)
        ) {
          try {
            await this.recover(account, generation);
          } catch {
            /* Retry recovery on the next connection. */
          }
        }
      }
      await pause(1500, signal);
    }
  }
  async send(text: string, replyTo?: NativeMessage, tokens: MentionToken[] = []): Promise<boolean> {
    const account = this.value.account,
      threadId = this.value.selected,
      generation = this.generation;
    if (!account || !threadId || !text.trim()) return false;
    let mentions: TextMentions | undefined;
    try { mentions = mentionOptions(text, tokens); }
    catch (error) { this.publish({ error: error instanceof Error ? error.message : t("session.chooseMentionedMembersAgain") }); return false; }
    if (mentions && (!threadId.endsWith("@chatroom") || this.value.capabilities.inlineTextMentions !== true)) {
      this.publish({ error: t("session.hubCannotSendInlineMentions") });
      return false;
    }
    if (mentions?.all && !canMentionAll(this.value.selectedGroup, threadId)) {
      this.publish({ error: allMentionProblem(this.value.selectedGroup, threadId) });
      return false;
    }
    if (this.value.selectedGroup?.isInGroup === false) {
      this.publish({
        error: t("session.noLongerGroupCachedHistory"),
      });
      return false;
    }
    const target =
      replyTo &&
      (this.value.messages.find((message) => message.id === replyTo.id) ??
        replyTo);
    if (
      target &&
      (replyTo?.accountId !== account ||
        replyTo?.threadId !== threadId ||
        target.accountId !== account ||
        target.threadId !== threadId ||
        target.serverId !== replyTo?.serverId ||
        !canReply(target) ||
        this.value.capabilities.quotedTextSend !== true)
    ) {
      this.publish({
        error: t("session.messageCanTQuotedChoose"),
      });
      return false;
    }
    const quote = target
      ? { messageId: target.id, serverId: target.serverId! }
      : undefined;
    const quotePreview = target
      ? messageSummary(
          target,
          target.direction === "outgoing"
            ? t("session.you")
            : this.value.contacts.find(
                (contact) => contact.id === target.senderId,
              )?.displayName ?? target.senderId,
        )
      : undefined;
    if (quotePreview && target?.type === 49 && target.appMessageType === 19)
      quotePreview.text = quotedText(target).replace(/\s+/g, " ").slice(0, 240);
    const key = crypto.randomUUID();
    this.publish({
      sends: [
        ...this.value.sends,
        { key, text, threadId, ...(quote && { quote, quotePreview }), ...(mentions && { mentions }) },
      ],
      error: null,
    });
    void this.readOnInteraction(account, threadId);
    try {
      const intent = await this.client.sendText(
        account,
        threadId,
        text,
        key,
        quote || mentions ? { ...(quote && { quote }), ...(mentions && { mentions }) } : undefined,
      );
      if (this.current(account, generation)) {
        await this.acceptReceipt(intent, account, generation, key);
        // 入队已确认，补查只是为了追上可能先到的事件。查询失败时保留回执，继续等事件更新。
        try {
          await this.acceptReceipt(
            await this.client.send(account, intent.id),
            account,
            generation,
            key,
          );
        } catch {
          // 后续的 send.updated 或手动查询会重新读取同一条发送记录。
        }
      } else {
        const sends = (this.pending.get(account) ?? []).map((send) =>
          send.key === key ? { ...send, intent } : send,
        );
        this.pending.set(account, sends);
        this.receipts?.write(account, sends);
      }
    } catch (error) {
      if (
        this.current(account, generation) &&
        error instanceof NativeApiError &&
        error.code === "auth_required"
      )
        this.signedOut(account);
      if (
        (quote || mentions) &&
        error instanceof NativeApiError &&
        !error.retryable &&
        [
          "invalid_quote_reference",
          "invalid_quote_request",
          "unsupported_quote_type",
          "quote_target_unavailable",
          "quote_target_mismatch",
          "quote_target_ambiguous",
          "quote_content_unavailable",
          "quote_sender_unavailable",
          "invalid_mentions", "invalid_mention_text", "mentions_require_group", "mention_member_unavailable", "mention_group_membership_unavailable", "mention_all_forbidden", "mention_all_permission_unavailable", "quote_mentions_unsupported", "invalid_request",
        ].includes(error.code)
      ) {
        // These admission errors occur before an intent exists. Preserve the
        // composer's text/reference instead of showing an uncertain send.
        if (this.current(account, generation)) {
          if (this.value.selected === threadId && ["mention_all_forbidden", "mention_all_permission_unavailable"].includes(error.code))
            this.invalidateAllMentionPermission();
          this.publish({
            sends: this.value.sends.filter((send) => send.key !== key),
            error: readableError(error),
          });
        } else {
          const sends = (this.pending.get(account) ?? []).filter(
            (send) => send.key !== key,
          );
          this.pending.set(account, sends);
          this.receipts?.write(account, sends);
        }
        return false;
      }
      const update = (sends: PendingSend[]) =>
        sends.map((send) =>
          send.key === key
            ? { ...send, uncertain: true, error: readableError(error) }
            : send,
        );
      if (this.current(account, generation))
        this.publish({ sends: update(this.value.sends) });
      else {
        const sends = update(this.pending.get(account) ?? []);
        this.pending.set(account, sends);
        this.receipts?.write(account, sends);
      }
    }
    return true;
  }
  async checkSend(key: string): Promise<void> {
    const account = this.value.account,
      generation = this.generation;
    if (!account) return;
    const known = this.value.sends.find((send) => send.key === key)?.intent;
    try {
      const receipt = known
        ? await this.client.send(account, known.id)
        : await this.client.sendByKey(account, key);
      if (this.value.sends.find((send) => send.key === key)?.intent !== known)
        return;
      if ("clientId" in receipt)
        await this.acceptReceipt(receipt, account, generation, key);
    } catch (error) {
      if (
        this.current(account, generation) &&
        this.value.sends.find((send) => send.key === key)?.intent === known
      )
        this.publish({
          sends: this.value.sends.map((send) =>
            send.key === key
              ? { ...send, uncertain: true, error: readableError(error) }
              : send,
          ),
        });
    }
  }
  async cancelSend(key: string): Promise<void> {
    const account = this.value.account,
      generation = this.generation,
      known = this.value.sends.find((send) => send.key === key)?.intent;
    if (!account || known?.state !== "queued") return;
    try {
      const receipt = await this.client.cancelSend(account, known.id);
      if ("clientId" in receipt)
        await this.acceptReceipt(receipt, account, generation, key);
    } catch (error) {
      if (
        this.current(account, generation) &&
        this.value.sends.find((send) => send.key === key)?.intent === known
      )
        this.publish({
          sends: this.value.sends.map((send) =>
            send.key === key
              ? { ...send, uncertain: true, error: readableError(error) }
              : send,
          ),
        });
    }
  }
  dismissSend(key: string): void {
    this.publish({
      sends: this.value.sends.map((send) =>
        send.key === key &&
        (send.uncertain ||
          ["failed", "rejected", "cancelled", "needs_review"].includes(
            send.intent?.state ?? "",
          ))
          ? { ...send, dismissed: true }
          : send,
      ),
    });
  }
}
