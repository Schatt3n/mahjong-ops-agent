"""首次启动时在回环地址收集模型配置。凭证只交给当前启动器。"""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import secrets
import subprocess
import sys
import time
from urllib.parse import parse_qs


HOST = "127.0.0.1"
SETUP_TIMEOUT_SECONDS = 600
MAX_POST_BYTES = 16 * 1024
FIELDS = {
    "provider": "MAHJONG_LLM_PROVIDER",
    "model": "MAHJONG_LLM_MODEL",
    "api_key": "MAHJONG_LLM_API_KEY",
    "base_url": "MAHJONG_LLM_BASE_URL",
    "api_protocol": "MAHJONG_LLM_API_PROTOCOL",
    "response_format": "MAHJONG_LLM_RESPONSE_FORMAT",
}
PROVIDERS = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "glm": "GLM",
    "zai": "Z.ai",
    "bigmodel": "BigModel / Z.ai",
    "opencode_go": "OpenCode Go",
    "openai_compatible": "自定义 OpenAI 兼容接口",
}
API_PROTOCOLS = {
    "chat_completions": "Chat Completions",
    "responses": "Responses",
}
RESPONSE_FORMATS = {
    "json_object": "JSON 对象",
    "text": "文本",
    "": "跟随接口默认值",
}

STYLE = """
    :root { color-scheme: light; font: 16px/1.5 -apple-system, BlinkMacSystemFont,
        "PingFang SC", "Microsoft YaHei", system-ui, sans-serif; color: #182523; background: #ecf1ef; }
    * { box-sizing: border-box; }
    body { min-height: 100vh; margin: 0; padding: 48px 20px; display: grid; place-items: center; }
    main { width: min(100%, 660px); padding: clamp(24px, 5vw, 44px); background: #fff;
        border: 1px solid #cbd9d5; border-radius: 14px; box-shadow: 0 12px 36px #183b3210; }
    h1 { font-size: clamp(1.65rem, 4vw, 2rem); line-height: 1.25; margin: 0 0 10px; }
    p { color: #49605a; margin: 0; }
    form { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 18px; margin-top: 26px; }
    label { display: grid; gap: 6px; align-content: start; font-weight: 600; min-width: 0; }
    input, select, button { font: inherit; }
    input, select { width: 100%; height: 44px; padding: 8px 11px; color: #182523;
        background: #fafcfb; border: 1px solid #9eada9; border-radius: 6px; }
    input:focus-visible, select:focus-visible, button:focus-visible { outline: 2px solid #087669; outline-offset: 2px; }
    small { font-size: .84rem; font-weight: 400; color: #49605a; }
    .wide, footer { grid-column: 1 / -1; }
    footer { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 10px; padding-top: 10px; }
    button { min-height: 44px; padding: 6px 16px; border: 1px solid #b4c3bf;
        border-radius: 6px; background: white; color: #1c3530; cursor: pointer; font-weight: 650; }
    .primary { border-color: #07564e; background: #07564e; color: #fff; }
    .error { margin-top: 20px; padding: 10px 12px; border: 1px solid #e2b9b0;
        border-radius: 6px; color: #7d3025; background: #fff4f1; overflow-wrap: anywhere; }
    @media (max-width: 540px) { body { padding: 16px; } form { grid-template-columns: 1fr; gap: 16px; } }
"""


class ModelSetupCancelled(Exception):
    """打开网页后用户取消配置或等待超时，可结束本次启动。"""


def _document(content: str) -> str:
    """生成单一页面外壳，浏览器无需加载任何外部脚本、字体或样式。"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>配置主 Agent 模型</title><style>{STYLE}</style></head>
<body><main>{content}</main></body></html>"""


def _options(choices: dict[str, str], value: str) -> str:
    """保留本机已有的供应商选项，同时把所有动态内容输出为 HTML 文本。"""
    if value not in choices:
        choices = {**choices, value: value}
    return "".join(
        f'<option value="{escape(key)}"{" selected" if key == value else ""}>{escape(label)}</option>'
        for key, label in choices.items()
    )


def _form(route: str, settings: dict[str, str], message: str = "") -> str:
    """已有模型和接口可预填，API key 在首次显示和校验失败时均为空。"""
    provider_value = settings.get(FIELDS["provider"]) or "deepseek"
    model_value = settings.get(FIELDS["model"], "")
    protocol = settings.get(FIELDS["api_protocol"])
    if not protocol:
        # 已有 Go Luna 模型 ID 足以选定官方端点。其他模型继续使用通用默认协议。
        protocol = "responses" if (provider_value, model_value) == ("opencode_go", "gpt-6-luna") else "chat_completions"
    provider = _options(PROVIDERS, provider_value)
    model = escape(model_value)
    base_url = escape(settings.get(FIELDS["base_url"], ""))
    api_protocol = _options(API_PROTOCOLS, protocol)
    response_format = _options(RESPONSE_FORMATS, settings.get(FIELDS["response_format"], "json_object"))
    error = f'<p class="error" role="alert">{escape(message)}</p>' if message else ""
    return _document(f"""
    <h1>配置主 Agent 模型</h1>
    <p>填入模型服务信息后，启动器会继续启动项目。</p>
    {error}
    <form method="post" action="{route}" autocomplete="off">
      <label>供应商<select name="provider" required>{provider}</select></label>
      <label>模型 ID<input name="model" value="{model}" placeholder="例如 deepseek-flash" maxlength="256"
          required spellcheck="false" autocapitalize="none"></label>
      <label class="wide">API key<input type="password" name="api_key" autocomplete="new-password"
          maxlength="4096" required spellcheck="false" autocapitalize="none"></label>
      <label class="wide">接口地址<input type="url" name="base_url" value="{base_url}" maxlength="2048"
          placeholder="https://api.example.com/v1" spellcheck="false" autocapitalize="none">
          <small>预设供应商可留空，自定义接口填写 Base URL。</small></label>
      <label>API 协议<select name="api_protocol" required>{api_protocol}</select>
          <small>OpenCode Go 的 GPT‑6 Luna 请选择 Responses。</small></label>
      <label>响应格式<select name="response_format">{response_format}</select>
          <small>接口拒绝指定格式时，选择「跟随接口默认值」。</small></label>
      <footer>
        <button class="primary" type="submit" name="action" value="save">保存并启动</button>
        <button type="submit" name="action" value="cancel" formnovalidate>取消</button>
      </footer>
    </form>""")


def _open_browser(url: str) -> bool:
    """使用系统桌面打开回环页面，无桌面或打开失败时让启动器回退。"""
    if sys.platform == "darwin":
        command = ["/usr/bin/open", url]
    elif os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"):
        command = ["xdg-open", url]
    else:
        return False
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return process.wait(timeout=2) == 0
    except subprocess.TimeoutExpired:
        # 部分桌面启动器会等到浏览器退出，先开始响应本机请求。
        return True
    except OSError:
        return False


def configure_model_in_browser(
    *, initial: dict[str, str], validate: Callable[[dict[str, str]], None],
) -> dict[str, str] | None:
    """等待一次本机表单提交；无桌面时返回空值，取消或超时则结束本次启动。

    回调只负责校验配置，可用 ValueError 提供中文错误。调用方负责持久化，
    也可用相同校验逻辑处理终端输入。
    """
    route = "/setup/" + secrets.token_urlsafe(32)
    result: dict[str, str] | None = None
    cancelled = False

    class SetupHandler(BaseHTTPRequestHandler):
        """只信任绑定地址与本次会话的路径，提交还需要同源校验。"""

        timeout = 2

        def log_message(self, *_args) -> None:
            """关闭默认访问日志，避免写入临时访问路径或请求内容。"""

        def _send_page(self, status: int, content: str) -> None:
            """禁止缓存、嵌入和第三方请求，不回传校验用的 API key。"""
            payload = content.encode("utf-8")
            self.send_response_only(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            # 浏览器需要为同源表单提供 Origin，同时不能向外部站点暴露会话路径。
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            try:
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _owns_route(self) -> bool:
            """拒绝其他 Host 和路径，防止外部域名借回环地址访问配置。"""
            return self.headers.get_all("Host") == [authority] and self.path == route

        def do_GET(self) -> None:
            """仅向当次启动打开的会话地址提供表单。"""
            if not self._owns_route():
                self._send_page(404, "页面不存在。")
                return
            self._send_page(200, _form(route, initial))

        def _read_post(self) -> dict[str, str]:
            """只接受有大小上限且字段唯一的普通表单，拒绝其他提交协议。"""
            if (
                self.headers.get_content_type() != "application/x-www-form-urlencoded"
                or self.headers.get("Transfer-Encoding")
            ):
                raise ValueError
            lengths = self.headers.get_all("Content-Length") or []
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise ValueError
            length = int(lengths[0])
            if not 0 < length <= MAX_POST_BYTES:
                raise ValueError
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError
            fields = parse_qs(
                body.decode(), keep_blank_values=True, strict_parsing=True,
                max_num_fields=len(FIELDS) + 1, errors="strict",
            )
            if fields.keys() != (FIELDS.keys() | {"action"}) or any(len(values) != 1 for values in fields.values()):
                raise ValueError
            return {name: values[0] for name, values in fields.items()}

        def do_POST(self) -> None:
            """仅在本机同源提交通过校验后，把值交回启动器并结束会话。"""
            nonlocal result, cancelled
            if not self._owns_route() or self.headers.get_all("Origin") != [origin]:
                self._send_page(403, "提交无效，请回到本机配置页面。")
                return
            try:
                fields = self._read_post()
            except (OSError, ValueError):
                self._send_page(400, "无法读取配置，请重新打开页面。")
                return
            action = fields["action"]
            if action == "cancel":
                cancelled = True
                self._send_page(200, _document("<h1>已取消模型配置</h1><p>可以关闭此页面。</p>"))
                return
            if action != "save":
                self._send_page(400, "提交方式无效。")
                return

            settings = {key: fields[name].strip() for name, key in FIELDS.items()}
            secret = settings[FIELDS["api_key"]]
            try:
                validate(settings)
            except ValueError as error:
                message = str(error) or "请检查模型配置。"
                if secret:
                    message = message.replace(secret, "[已隐藏]")
                self._send_page(422, _form(route, settings, message))
                return
            except Exception:
                # 回调异常可能携带输入凭证，不能交给 HTTPServer 输出 traceback。
                self._send_page(422, _form(route, settings, "无法校验配置，请检查设置后再试。"))
                return
            result = settings
            self._send_page(200, _document("<h1>已提交模型配置</h1><p>可以关闭此页面，启动进度会显示在启动窗口。</p>"))

    with HTTPServer((HOST, 0), SetupHandler) as server:
        authority = f"{HOST}:{server.server_port}"
        origin = f"http://{authority}"
        if not _open_browser(origin + route):
            return None
        server.timeout = 0.25
        deadline = time.monotonic() + SETUP_TIMEOUT_SECONDS
        while result is None and not cancelled and time.monotonic() < deadline:
            server.handle_request()
    if cancelled:
        raise ModelSetupCancelled("已取消模型配置。需要配置时可重新启动项目。")
    if result is None:
        raise ModelSetupCancelled("模型配置页面等待超时。请重新启动项目。")
    return result
