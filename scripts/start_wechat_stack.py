#!/usr/bin/env python3
"""串联本机 agent-wechat、主 Agent 和通道适配器的启动及回收。"""

from __future__ import annotations

import fcntl
import getpass
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from threading import Thread
import time
from urllib.parse import urlsplit
import warnings

if __package__:
    from .check_agent_wechat_update import check_for_update
else:
    from check_agent_wechat_update import check_for_update

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient, default_base_url  # noqa: E402

HOST = "127.0.0.1"
ADAPTER = "integrations/agent-wechat-adapter/src/main.mjs"
MODEL_ENV_NAME = ".env.wechat-model.local"
MODEL_KEY_NAMES = ("MAHJONG_LLM_API_KEY", "MAHJONG_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")


class LaunchError(Exception):
    """启动前置条件或组件就绪检查失败，错误消息可直接展示。"""


def model_is_configured() -> bool:
    """复用 Runtime 的配置工厂检查主模型与备用模型设置，不发起模型请求。"""
    try:
        return OpenAICompatibleAgentClient.from_env() is not None
    except ValueError:
        raise LaunchError(f"主 Agent 模型配置无效，请检查 .env 和 {MODEL_ENV_NAME}。") from None


def require_runtime_model(path: Path) -> None:
    """首次启动时补齐 DeepSeek 模型设置，凭证只保存到私有的本机文件。"""
    if model_is_configured():
        return
    explicit_key_name = next((name for name in MODEL_KEY_NAMES if os.getenv(name)), None)
    provider = os.getenv("MAHJONG_LLM_PROVIDER", "").strip().lower()
    endpoint = os.getenv("MAHJONG_LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("MAHJONG_LLM_MODEL", "").strip()
    if explicit_key_name == MODEL_KEY_NAMES[0] and not (provider or endpoint or model.startswith("deepseek-")):
        raise LaunchError("请在 .env 中补齐现有 API key 对应的 MAHJONG_LLM_PROVIDER 和 MAHJONG_LLM_MODEL。")

    local_settings = load_dotenv_defaults(path)
    if explicit_key_name and MODEL_KEY_NAMES[0] in local_settings:
        # 已配置的 DeepSeek key 别名与显式主 key 具有相同的来源优先级。
        os.environ.pop(MODEL_KEY_NAMES[0])
    provider = os.getenv("MAHJONG_LLM_PROVIDER", "").strip().lower()
    endpoint = os.getenv("MAHJONG_LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("MAHJONG_LLM_MODEL", "").strip()
    deepseek_url = default_base_url("deepseek")
    # 已保存的 DeepSeek key 只用于 DeepSeek 官方接口，不能填入其他供应商的空配置。
    if (provider and provider != "deepseek") or (endpoint and endpoint not in {deepseek_url, deepseek_url + "/v1"}) or (model and not model.startswith("deepseek-")):
        raise LaunchError("请在 .env 中补齐当前模型的 MAHJONG_LLM_API_KEY 和 MAHJONG_LLM_MODEL。")
    defaults = {
        "MAHJONG_LLM_PROVIDER": "deepseek",
        "MAHJONG_LLM_MODEL": model or "deepseek-flash",
    }
    key = next((os.environ[name] for name in MODEL_KEY_NAMES if os.getenv(name)), None)
    if not key:
        if path.exists():
            raise LaunchError(f"{MODEL_ENV_NAME} 中缺少 DeepSeek API key，请补齐或在 .env 中配置模型。")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("首次启动，请输入 DeepSeek API key（输入不会显示） ").strip()
        except (EOFError, getpass.GetPassWarning):
            raise LaunchError("请在终端启动并输入 DeepSeek API key，或先在 .env 中配置模型。") from None
        if not key or any(not char.isprintable() or char.isspace() or char in "\"'" for char in key):
            raise LaunchError("DeepSeek API key 为空或格式有误，请重新启动后输入。")
        defaults["MAHJONG_LLM_API_KEY"] = key

    os.environ.update(defaults)
    if not model_is_configured():
        raise LaunchError("缺少主 Agent 模型配置，请检查 .env。")
    if "MAHJONG_LLM_API_KEY" in defaults:
        # 文件写完才发布，避免中断留下半份凭证，也不会覆盖已有的配置文件。
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f"{path.name}.", encoding="utf-8") as file:
            file.writelines(f"{name}={value}\n" for name, value in defaults.items())
            file.flush()
            os.link(file.name, path)
        print(f"已保存本机模型配置到 {MODEL_ENV_NAME}。", flush=True)


def bootstrap_tool(name: str) -> str:
    """调用项目的依赖安装入口，只用其标准输出提供的可执行文件路径。"""
    try:
        path = subprocess.check_output(["bash", str(ROOT / "scripts/bootstrap_wechat_tool.sh"), name], text=True).strip()
        if path and os.access(path, os.X_OK):
            return path
    except (OSError, subprocess.SubprocessError):
        pass
    raise LaunchError(f"{name} 安装失败，请查看上方的依赖安装信息。")


def resolve_node() -> str:
    """优先使用已有的 Node.js 20+，缺失时由项目安装；显式指定的路径需要有效。"""
    configured = os.getenv("MAHJONG_WECHAT_NODE_BIN")
    candidate = os.path.expanduser(configured) if configured else shutil.which("node")
    if candidate:
        try:
            version = subprocess.check_output([candidate, "--version"], text=True, timeout=5).strip()
            if int(version.lstrip("v").split(".")[0]) >= 20:
                return candidate
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if configured:
        raise LaunchError("MAHJONG_WECHAT_NODE_BIN 必须指向可用的 Node.js 20 或更高版本。")
    return bootstrap_tool("node")


def local_path(key: str, default: Path) -> Path:
    """把本机配置里的相对路径和用户目录路径统一交给子进程。"""
    return Path(os.getenv(key) or default).expanduser().resolve()


def read_token(path: Path) -> str:
    """读取本机令牌，允许负责生成它的服务尚未启动。"""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def get_json(port: int, route: str, token: str = "") -> tuple[int, dict]:
    """直连回环接口，令牌不会经过代理或被重定向。"""
    conn = http.client.HTTPConnection(HOST, port, timeout=2)
    try:
        conn.request("GET", route, headers={"Authorization": f"Bearer {token}"} if token else {})
        with conn.getresponse() as response:
            status = response.status
            data = json.loads(response.read()) if status == 200 else {}
        return status, data if isinstance(data, dict) else {}
    except (OSError, ValueError, http.client.HTTPException):
        return 0, {}
    finally:
        conn.close()


def existing_listener(port: int, route: str, expected: tuple[str, str]) -> dict:
    """只接入符合项目健康接口的服务，端口冲突时保留原进程。"""
    try:
        with socket.create_connection((HOST, port), timeout=1):
            pass
    except ConnectionRefusedError:
        return {}
    _, health = get_json(port, route)
    field, value = expected
    if health.get(field) != value:
        raise LaunchError(f"端口 {port} 已占用，健康检查与预期服务不符。")
    return health


def select_account(accounts: list[dict], configured: str) -> dict | None:
    """空注册表可创建账号，多账号时只接受唯一匹配的 ID 或别名。"""
    if not accounts and not configured:
        return None
    if not configured and len(accounts) == 1:
        return accounts[0]
    matches = [item for item in accounts if configured and configured in (item.get("id"), item.get("alias"))]
    if len(matches) != 1:
        raise LaunchError("请将 MAHJONG_AGENT_WECHAT_ACCOUNT 设为唯一的 agent-wechat 账号 ID 或别名。")
    return matches[0]


def is_online(account: dict) -> bool:
    """连接已经在线且会话已通过认证后，适配器才可以固定账号和游标。"""
    return account.get("authenticated") is True and account.get("connection") == "online"


def show_qr(image: Path) -> None:
    """仅在需要扫码时打开二维码图片，无法打开时保留终端路径。"""
    if sys.platform == "darwin":
        command = ["open", "-a", "Preview", str(image)]
    else:
        command = ["xdg-open", str(image)]
    try:
        subprocess.run(command, check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        print(f"请手动打开 {image}", flush=True)


def start_update_check(port: int, token: str) -> Thread | None:
    """按当前运行的 daemon 版本判断是否有新发布，查询故障不影响启动。"""
    if os.getenv("MAHJONG_WECHAT_UPDATE_CHECK", "1") == "0":
        return None
    _, status = get_json(port, "/v1/status", token)
    version = status.get("version")
    if not isinstance(version, dict) or version.get("product") != "agent-wechat":
        return None
    current = version.get("version")
    if not isinstance(current, str):
        return None
    cache = ROOT / "runtime_data/wechat-stack-tools/agent-wechat-release.tag"
    worker = Thread(target=check_for_update, args=(current, cache), daemon=True)
    worker.start()
    return worker


class Stack:
    """只管理本次创建的进程；复用的服务由原启动方管理。"""

    def __init__(self) -> None:
        self.children: dict[str, subprocess.Popen] = {}

    def spawn(
        self, name: str, command: list[str], *, quiet: bool = False, private_input: bytes | None = None,
    ) -> subprocess.Popen:
        """每个子服务使用单独的进程组，方便回收 npm 等启动包装器。"""
        child = subprocess.Popen(
            command, cwd=ROOT, start_new_session=True,
            stdin=subprocess.PIPE if private_input is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL if quiet else None,
        )
        self.children[name] = child
        if child.stdin is not None:
            with child.stdin as pipe:
                pipe.write(private_input)
        return child

    def check_children(self) -> None:
        """就绪等待和运行期间都检测提前退出，避免遗留半套服务。"""
        for name, child in self.children.items():
            if child.poll() is not None:
                raise LaunchError(f"{name} 已退出（状态 {child.returncode}），请查看上方输出。")

    def wait_ready(self, name: str, check):
        """等待健康接口或登录状态，避免端口刚打开就报告已启动。"""
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            self.check_children()
            result = check()
            if result:
                return result
            time.sleep(0.25)
        raise LaunchError(f"等待 {name} 就绪超时。")

    def close(self) -> None:
        """先停适配器和 Runtime，再停 daemon，给 Runtime 和扫码流程留出清理时间。"""
        for name, child in reversed(self.children.items()):
            if child.poll() is not None:
                continue
            try:
                request_signal = signal.SIGINT if name in {"主 Agent", "微信登录"} else signal.SIGTERM
                os.killpg(child.pid, request_signal)
                child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()

    def login(self, command: list[str]) -> None:
        """优先使用上游保存的登录信任，仅显示上游实际生成的二维码。"""
        print("正在恢复 agent-wechat 登录。如需确认，请查看手机微信。", flush=True)
        with tempfile.TemporaryDirectory(prefix="mahjong-wechat-qr-") as directory:
            path = Path(directory) / "login.png"
            child = self.spawn("微信登录", [*command, "auth", "login", "--qr-output", str(path)])
            versions: set[Path] = set()
            try:
                while child.poll() is None:
                    # wx 自动更新过期二维码，每个版本写入不同文件。
                    images = {image for image in path.parent.glob("login*.png") if image.stat().st_size}
                    fresh = images - versions
                    if fresh:
                        latest = max(fresh, key=lambda image: image.stat().st_mtime_ns)
                        print("请用手机微信扫描打开的二维码并确认登录。", flush=True)
                        if os.getenv("MAHJONG_WECHAT_OPEN_QR", "1") != "0":
                            show_qr(latest)
                        versions = images
                    for name, service in self.children.items():
                        if name != "微信登录" and service.poll() is not None:
                            raise LaunchError(f"{name} 在微信登录期间退出。")
                    time.sleep(0.25)
                if child.returncode:
                    raise LaunchError("agent-wechat 登录未完成，请查看上方信息后重试。")
            finally:
                # 登录子进程若仍在等待，交给 close() 发送 SIGINT 取消尝试。
                if child.poll() is not None:
                    self.children.pop("微信登录")


def start(stack: Stack) -> None:
    """校验回环配置，并按账号在线、Runtime 就绪、适配器就绪的顺序启动。"""
    upstream = urlsplit(os.getenv("MAHJONG_AGENT_WECHAT_BASE_URL") or f"http://{HOST}:6174")
    if upstream.scheme != "http" or upstream.netloc != f"{HOST}:{upstream.port}" or upstream.path not in {"", "/"} or upstream.query or upstream.fragment:
        raise LaunchError("MAHJONG_AGENT_WECHAT_BASE_URL 必须是 http://127.0.0.1:端口。")
    wx_port = upstream.port
    runtime_port = int(os.getenv("MAHJONG_AGENT_PORT") or 8790)
    adapter_port = int(os.getenv("MAHJONG_AGENT_WECHAT_CONTROL_PORT") or 8792)
    if not wx_port or len({wx_port, runtime_port, adapter_port}) != 3 or not all(
        0 < port < 65536 for port in (wx_port, runtime_port, adapter_port)
    ):
        raise LaunchError("三个本机服务需配置各自独立的有效端口。")

    wx_url = f"http://{HOST}:{wx_port}"
    runtime_url = f"http://{HOST}:{runtime_port}"
    wx_existing = existing_listener(wx_port, "/health", ("status", "ok"))
    runtime_existing = existing_listener(runtime_port, "/api/health", ("runtime", "mahjong_agent_runtime"))
    adapter_existing = existing_listener(adapter_port, "/health", ("adapter", "agent-wechat"))
    if adapter_existing and (not wx_existing or not runtime_existing or adapter_existing.get("upstream_base_url") != wx_url):
        raise LaunchError("运行中的适配器与本次 Runtime 或 agent-wechat 配置不匹配。")

    xdg_data = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
    data_dir = local_path("MAHJONG_WECHAT_DATA_DIR", xdg_data / "agent-wechat")
    wx_token_path = local_path("MAHJONG_AGENT_WECHAT_TOKEN_FILE", data_dir / "api-token")
    runtime_token_path = local_path("MAHJONG_AGENT_API_TOKEN_PATH", ROOT / "data/runtime_api_token")
    wx_token = os.getenv("MAHJONG_AGENT_WECHAT_TOKEN", "").strip()
    runtime_token = os.getenv("MAHJONG_AGENT_API_TOKEN", "").strip()
    wx_bin = os.getenv("MAHJONG_WECHAT_WX_BIN") or shutil.which("wx")
    if not wx_bin:
        sibling_wx = ROOT.parent / "agent-wechat/packages/native-client-rust/target/release/wx"
        if sibling_wx.is_file() and os.access(sibling_wx, os.X_OK):
            wx_bin = str(sibling_wx)
    if wx_bin:
        wx_bin = os.path.expanduser(wx_bin)
    node = resolve_node() if not adapter_existing else None
    if not wx_existing and not wx_bin:
        wx_bin = bootstrap_tool("wx")
    if not runtime_existing:
        require_runtime_model(ROOT / MODEL_ENV_NAME)

    if not wx_existing:
        args = [wx_bin, "serve", "--data-dir", str(data_dir), "--listen", f"{HOST}:{wx_port}"]
        server_config = os.getenv("MAHJONG_WECHAT_SERVER_CONFIG")
        if server_config:
            args.extend(["--server-config", str(Path(server_config).expanduser())])
        if wx_token:
            args.append("--token-stdin")
        elif wx_token_path != data_dir / "api-token" or wx_token_path.is_file():
            args.extend(["--token-file", str(wx_token_path)])
        stack.spawn("agent-wechat", args, private_input=(wx_token + "\n").encode() if wx_token else None)
        stack.wait_ready("agent-wechat", lambda: get_json(wx_port, "/health")[1].get("status") == "ok")
    wx_token = wx_token or read_token(wx_token_path)
    if not wx_token:
        raise LaunchError("找不到 agent-wechat API 令牌，请核对 MAHJONG_AGENT_WECHAT_TOKEN_FILE。")
    os.environ["AGENT_WECHAT_TOKEN"] = wx_token
    update_check = start_update_check(wx_port, wx_token)

    def accounts() -> list[dict]:
        """每次从守护进程读取账号状态，禁止用旧的登录结果启动适配器。"""
        status, payload = get_json(wx_port, "/v1/accounts", wx_token)
        items = payload.get("items")
        if status != 200 or not isinstance(items, list):
            raise LaunchError(f"无法读取 agent-wechat 账号（HTTP {status}），请核对服务地址和 API 令牌。")
        return items

    if runtime_existing and not verify_runtime(runtime_port, runtime_token_path, runtime_token):
        raise LaunchError("运行中的主 Agent 未接受本机配置的 API 令牌。")

    selected = os.getenv("MAHJONG_AGENT_WECHAT_ACCOUNT", "").strip()
    account = select_account(accounts(), selected)
    if adapter_existing and (not account or adapter_existing.get("account_id") != account.get("id")):
        raise LaunchError("运行中的适配器使用了其他账号。")
    if account and account.get("connection") == "reconnecting":
        deadline = time.monotonic() + 10
        while account.get("connection") == "reconnecting" and time.monotonic() < deadline:
            stack.check_children()
            time.sleep(0.5)
            account = select_account(accounts(), account["id"])
            if not account:
                raise LaunchError("等待重连时，所选的微信账号已不存在。")
    if not account or not is_online(account):
        if not wx_bin:
            wx_bin = bootstrap_tool("wx")
        command = [wx_bin, "--server", wx_url]
        if not account:
            alias = "mahjong-work"
            created = stack.spawn("创建微信账号", [*command, "accounts", "create", "--alias", alias], quiet=True)
            if created.wait(timeout=15):
                raise LaunchError("agent-wechat 账号创建失败。")
            stack.children.pop("创建微信账号")
            account = select_account(accounts(), alias)
            if not account:
                raise LaunchError("新建 agent-wechat 账号尚未注册。")
        account_id = account["id"]
        stack.login([*command, "--account", account_id])
        stack.wait_ready("微信账号", lambda: any(item.get("id") == account_id and is_online(item) for item in accounts()))

    account_id = account["id"]
    print("agent-wechat 已在线。", flush=True)
    os.environ.update({
        "MAHJONG_AGENT_PORT": str(runtime_port),
        "MAHJONG_AGENT_API_TOKEN_PATH": str(runtime_token_path),
        "MAHJONG_AGENT_WECHAT_TOKEN": wx_token,
        "MAHJONG_AGENT_WECHAT_BASE_URL": wx_url,
        "MAHJONG_AGENT_WECHAT_ACCOUNT": account_id,
        "MAHJONG_AGENT_WECHAT_RAW_ENDPOINT": f"{runtime_url}/api/channels/agent-wechat/raw",
        "MAHJONG_AGENT_WECHAT_CONTROL_HOST": HOST,
        "MAHJONG_AGENT_WECHAT_CONTROL_PORT": str(adapter_port),
        "MAHJONG_WECHATY_OUTBOUND_BASE_URL": f"http://{HOST}:{adapter_port}",
    })
    if not runtime_existing:
        control_token = os.getenv("MAHJONG_AGENT_WECHAT_CONTROL_TOKEN")
        if control_token:
            os.environ.setdefault("MAHJONG_WECHATY_OUTBOUND_TOKEN", control_token)
        stack.spawn("主 Agent", [sys.executable, "-u", "scripts/run_agent_app.py"])
        stack.wait_ready("主 Agent", lambda: get_json(runtime_port, "/api/health")[1].get("ok") is True)
        if not verify_runtime(runtime_port, runtime_token_path, runtime_token):
            raise LaunchError("主 Agent 的 API 令牌验证失败。")
    if not adapter_existing:
        stack.spawn("微信适配器", [node, ADAPTER])
        stack.wait_ready("微信适配器", lambda: get_json(adapter_port, "/health")[1].get("account_id") == account_id)

    print(f"项目已就绪。控制台 {runtime_url}/", flush=True)
    if stack.children:
        print("按 Ctrl-C 停止本次启动的服务。", flush=True)
    elif update_check is not None:
        # 复用现有整套服务时，留出短时间显示更新提示，再退出启动任务。
        update_check.join(timeout=4)
    while stack.children:
        stack.check_children()
        time.sleep(0.5)


def verify_runtime(port: int, token_path: Path, explicit_token: str) -> bool:
    """使用适配器将要用的令牌校验同一个 Runtime，首次请求可生成默认令牌。"""
    token = explicit_token or read_token(token_path)
    if not token:
        get_json(port, "/api/runtime")
        token = read_token(token_path)
    if os.getenv("MAHJONG_AGENT_WECHAT_AGENT_TOKEN", "").strip():
        token = os.environ["MAHJONG_AGENT_WECHAT_AGENT_TOKEN"].strip()
    return bool(token) and get_json(port, "/api/runtime", token)[0] == 200


def main() -> int:
    """加载项目配置、持有启动锁，并在中断或失败时回收本次启动的服务。"""
    os.chdir(ROOT)
    os.umask(0o077)
    load_dotenv_defaults(ROOT / ".env.wechat-stack.local")
    load_dotenv_defaults(ROOT / ".env")
    lock_path = ROOT / "data/wechat-stack.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stack = Stack()

    def stop(_signum, _frame):
        """让 Codex 停止和终端 Ctrl-C 走同一条清理路径。"""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有一个项目启动入口正在运行。", flush=True)
            return 0
        try:
            start(stack)
            return 0
        except KeyboardInterrupt:
            return 130
        except (LaunchError, OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"启动失败：{error}", file=sys.stderr, flush=True)
            return 1
        finally:
            stack.close()


if __name__ == "__main__":
    raise SystemExit(main())
