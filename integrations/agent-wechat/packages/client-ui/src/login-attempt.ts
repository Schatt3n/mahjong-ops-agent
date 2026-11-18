import {
  NativeApiError,
  type NativeOperation,
  type NativeWeChatClient,
} from "@agent-wechat/shared";

export type SavedLogin = { key: string; operationId?: string; computerName?: string };
export type LoginStorage = {
  read: () => SavedLogin | null;
  write: (value: SavedLogin | null) => void;
};
const terminal = new Set([
  "succeeded",
  "failed",
  "expired",
  "rejected",
  "cancelled",
  "needs_login",
]);

/** 保存登录请求和设备显示名，确保重试仍使用同一份幂等请求。 */
export class LoginAttempt {
  private client: NativeWeChatClient;
  private account: string;
  private storage: LoginStorage;
  private computerName?: string;
  constructor(
    client: NativeWeChatClient,
    account: string,
    storage: LoginStorage,
    computerName?: string,
  ) {
    this.client = client;
    this.account = account;
    this.storage = storage;
    this.computerName = computerName;
  }
  async open(): Promise<NativeOperation> {
    let saved = this.storage.read();
    if (saved?.operationId) {
      try {
        const operation = await this.client.operation(
          this.account,
          saved.operationId,
        );
        if (operation.accountId !== this.account || operation.kind !== "login")
          throw new Error("Invalid login operation identity");
        if (!terminal.has(operation.state) || operation.state === "succeeded")
          return operation;
      } catch (error) {
        if (!(error instanceof NativeApiError) || error.status !== 404)
          throw error;
      }
      this.storage.write(null);
      saved = null;
    }
    // Hub 会校验幂等键对应的请求体。即使 Mac 更名，或旧版请求未带名称，重试也须沿用原请求。
    const request = saved ?? {
      key: crypto.randomUUID(),
      computerName: this.computerName,
    };
    const { key, computerName } = request;
    this.storage.write({ key, computerName });
    const operation = await this.client.login(
      this.account,
      {
        strategy: "auto",
        timeoutSeconds: 900,
        qrRefresh: { mode: "auto", maxRefreshes: 3 },
        ...(computerName === undefined ? {} : { computerName }),
      },
      key,
    );
    this.storage.write({ key, computerName, operationId: operation.id });
    return operation;
  }
  complete(operationId: string): void {
    if (this.storage.read()?.operationId === operationId)
      this.storage.write(null);
  }
}
