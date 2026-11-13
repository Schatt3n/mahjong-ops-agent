/** Only configured preview routes receive credentials; the browser stays on its UI origin. */
export function previewFetch(endpoint: string, origin: string, proxyPath: string, transport: typeof fetch = globalThis.fetch.bind(globalThis)): typeof fetch {
  if (!/^\/__hubs\/[a-z0-9-]{1,48}$/.test(proxyPath)) throw Error("Invalid preview proxy");
  const expected = new URL(endpoint).origin;
  const preview = new URL(origin).origin;
  return (input, init) => {
    const request = new Request(input, init);
    const url = new URL(request.url);
    if (url.origin !== expected || !url.pathname.startsWith("/v1/")) throw Error("Unexpected preview hub request");
    return transport(new Request(`${preview}${proxyPath}${url.pathname}${url.search}`, request));
  };
}
