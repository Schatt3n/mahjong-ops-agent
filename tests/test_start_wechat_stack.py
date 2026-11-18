from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from threading import Thread
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher_with_clean_model_env(monkeypatch):
    from scripts import start_wechat_stack as launcher

    monkeypatch.setattr(os, "environ", {
        key: value for key, value in os.environ.items()
        if not key.startswith(("MAHJONG_LLM_", "MAHJONG_DEEPSEEK_", "DEEPSEEK_"))
    })
    return launcher


def test_first_run_prompts_and_reuses_private_model_config(tmp_path, monkeypatch, capsys, launcher_with_clean_model_env):
    launcher = launcher_with_clean_model_env
    local = tmp_path / launcher.MODEL_ENV_NAME
    existing = tmp_path / ".env"
    existing.write_text("MAHJONG_LLM_MODEL=deepseek-reasoner\n", encoding="utf-8")
    launcher.load_dotenv_defaults(existing)
    monkeypatch.setattr(launcher.getpass, "getpass", lambda _: "test-deepseek-key")

    launcher.require_runtime_model(local)

    assert launcher.model_is_configured()
    assert local.stat().st_mode & 0o777 == 0o600
    assert "MAHJONG_LLM_API_KEY=test-deepseek-key\n" in local.read_text(encoding="utf-8")
    assert "test-deepseek-key" not in capsys.readouterr().out
    assert existing.read_text(encoding="utf-8") == "MAHJONG_LLM_MODEL=deepseek-reasoner\n"

    os.environ.pop("MAHJONG_LLM_API_KEY")
    os.environ["MAHJONG_LLM_MODEL"] = "deepseek-flash"
    os.environ["MAHJONG_LLM_PROVIDER"] = ""
    monkeypatch.setattr(launcher.getpass, "getpass", lambda _: pytest.fail("Existing key should be reused"))
    launcher.require_runtime_model(local)
    config = launcher.OpenAICompatibleAgentClient.from_env().config
    assert config.api_key == "test-deepseek-key"
    assert config.model == "deepseek-flash"
    assert config.base_url == "https://api.deepseek.com"


def test_configured_model_alias_needs_no_prompt(tmp_path, monkeypatch, launcher_with_clean_model_env):
    launcher = launcher_with_clean_model_env
    local = tmp_path / launcher.MODEL_ENV_NAME
    os.environ.update({
        "MAHJONG_LLM_MODEL": "deepseek-flash",
        "MAHJONG_LLM_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "test-existing-model-key",
    })
    monkeypatch.setattr(launcher.getpass, "getpass", lambda _: pytest.fail("Configured model should be reused"))

    launcher.require_runtime_model(local)

    assert not local.exists()


def test_missing_secure_input_does_not_save_key(tmp_path, monkeypatch, launcher_with_clean_model_env):
    launcher = launcher_with_clean_model_env
    local = tmp_path / launcher.MODEL_ENV_NAME

    def unavailable(_):
        raise launcher.getpass.GetPassWarning

    monkeypatch.setattr(launcher.getpass, "getpass", unavailable)
    with pytest.raises(launcher.LaunchError, match="终端"):
        launcher.require_runtime_model(local)
    assert not local.exists()


@contextmanager
def mock_wechat(online: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                payload = {"status": "ok"}
            elif self.headers.get("Authorization") != "Bearer test-wechat-token":
                self.send_error(401)
                return
            elif self.path == "/v1/status":
                payload = {"state": "ready", "version": {"version": "2.6.0", "product": "agent-wechat"}}
            elif self.path == "/v1/accounts":
                connected = online.exists()
                payload = {"items": [{
                    "id": "test-account",
                    "alias": "test-shop",
                    "ownUsername": "wxid_test" if connected else "",
                    "connection": "online" if connected else "disconnected",
                    "authenticated": connected,
                }]}
            elif self.path.startswith("/v1/accounts/test-account/events/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": connected\n\n")
                return
            elif self.path.startswith("/v1/accounts/test-account/events?"):
                payload = {"items": [], "nextCursor": "0"}
            else:
                self.send_error(404)
                return
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            thread.join()


def available_ports() -> tuple[int, int]:
    with socket.socket() as runtime, socket.socket() as adapter:
        runtime.bind(("127.0.0.1", 0))
        adapter.bind(("127.0.0.1", 0))
        return runtime.getsockname()[1], adapter.getsockname()[1]


def get_health(port: int, path: str) -> dict | None:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return json.loads(response.read()) if response.status == 200 else None
    except (OSError, http.client.HTTPException):
        return None
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("online_at_start", "bootstrap"), [(True, False), (False, False), (False, True)],
    ids=["saved-login", "qr-login", "missing-runtime-tools"],
)
def test_stack_launches_runtime_and_adapter_with_local_wechat(tmp_path, online_at_start, bootstrap):
    node = shutil.which("node")
    if node is None or int(subprocess.check_output([node, "--version"], text=True).split(".")[0][1:]) < 20:
        pytest.skip("Node.js 20 or newer is required by the adapter")

    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "scripts", checkout / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (checkout / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (checkout / "integrations").symlink_to(ROOT / "integrations", target_is_directory=True)
    (checkout / "pyproject.toml").symlink_to(ROOT / "pyproject.toml")
    (checkout / ".env").write_text(
        "MAHJONG_LLM_API_KEY=test-llm-key\nMAHJONG_LLM_MODEL=test-llm\n"
        "MAHJONG_LLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    token_file = tmp_path / "hub" / "api-token"
    token_file.parent.mkdir()
    token_file.write_text("test-wechat-token", encoding="utf-8")
    token_file.chmod(0o600)
    online = tmp_path / "wechat-online"
    qr_path_receipt = tmp_path / "qr-path"
    if online_at_start:
        online.touch()

    wx = tmp_path / "wx"
    wx.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "if '--qr-output' not in sys.argv:\n"
        "    sys.exit('logged-in daemon should not require another wx command')\n"
        "assert sys.argv[sys.argv.index('--account') + 1] == 'test-account'\n"
        "qr = Path(sys.argv[sys.argv.index('--qr-output') + 1])\n"
        "qr.write_bytes(b'\\x89PNG\\r\\n\\x1a\\nmock-qr')\n"
        f"online = Path({str(online)!r})\n"
        "for _ in range(100):\n"
        "    if online.exists():\n"
        "        print(json.dumps({'state': 'succeeded'}))\n"
        "        sys.exit(0)\n"
        "    time.sleep(0.1)\n"
        "sys.exit('test did not simulate scanning the QR code')\n",
        encoding="utf-8",
    )
    wx.chmod(0o700)
    viewer = tmp_path / ("open" if sys.platform == "darwin" else "xdg-open")
    viewer.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        "if sys.platform == 'darwin':\n"
        "    assert sys.argv[1:3] == ['-a', 'Preview']\n"
        f"receipt = Path({str(qr_path_receipt)!r})\n"
        "pending_receipt = receipt.with_suffix('.tmp')\n"
        "pending_receipt.write_text(sys.argv[-1])\n"
        "pending_receipt.replace(receipt)\n",
        encoding="utf-8",
    )
    viewer.chmod(0o700)
    bootstrap_log = tmp_path / "bootstrap-tools"
    if bootstrap:
        system_bin = tmp_path / "system-bin"
        system_bin.mkdir()
        (system_bin / "bash").symlink_to(shutil.which("bash"))
        (system_bin / viewer.name).symlink_to(viewer)
        (checkout / "scripts/bootstrap_wechat_tool.sh").write_text(
            "#!/bin/bash\n"
            f"printf '%s\\n' \"$1\" >> {shlex.quote(str(bootstrap_log))}\n"
            f"case \"$1\" in node) printf '%s\\n' {shlex.quote(node)} ;; "
            f"wx) printf '%s\\n' {shlex.quote(str(wx))} ;; *) exit 1 ;; esac\n",
            encoding="utf-8",
        )

    runtime_port, adapter_port = available_ports()
    with mock_wechat(online) as wx_port:
        env = {
            "HOME": str(tmp_path),
            "PATH": os.pathsep.join([str(tmp_path), str(Path(node).parent), str(Path(sys.executable).parent), os.defpath]),
            "TMPDIR": str(tmp_path),
            "MAHJONG_WECHAT_WX_BIN": str(wx),
            "MAHJONG_WECHAT_DATA_DIR": str(token_file.parent),
            "MAHJONG_AGENT_WECHAT_BASE_URL": f"http://127.0.0.1:{wx_port}",
            "MAHJONG_AGENT_WECHAT_TOKEN_FILE": str(token_file),
            "MAHJONG_AGENT_WECHAT_ACCOUNT": "test-shop",
            "MAHJONG_AGENT_WECHAT_STATE_DIR": str(tmp_path / "adapter-state"),
            "MAHJONG_AGENT_PORT": str(runtime_port),
            "MAHJONG_AGENT_WECHAT_CONTROL_PORT": str(adapter_port),
            "MAHJONG_WECHAT_UPDATE_CHECK": "0",
            "MAHJONG_WECHAT_OPEN_QR": "1",
        }
        if bootstrap:
            env["PATH"] = str(system_bin)
            env.pop("MAHJONG_WECHAT_WX_BIN")
        log = tmp_path / "launcher.log"
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [sys.executable, str(checkout / "scripts" / "start_wechat_stack.py")],
                cwd=checkout,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 12
                adapter = None
                qr = None
                while time.monotonic() < deadline and process.poll() is None:
                    if not online_at_start and qr is None and qr_path_receipt.exists():
                        qr = Path(qr_path_receipt.read_text(encoding="utf-8"))
                        assert qr.read_bytes().startswith(b"\x89PNG")
                        assert get_health(adapter_port, "/health") is None
                        online.touch()
                    adapter = get_health(adapter_port, "/health")
                    if adapter:
                        break
                    time.sleep(0.05)

                assert adapter, log.read_text(encoding="utf-8")
                assert adapter["account_id"] == "test-account"
                assert adapter["send_channel_enabled"] is False
                assert adapter["auto_send_reply"] is False
                assert qr_path_receipt.exists() is not online_at_start
                if bootstrap:
                    assert bootstrap_log.read_text(encoding="utf-8").splitlines() == ["node", "wx"]
                if qr is not None:
                    assert not qr.parent.exists()
                assert get_health(runtime_port, "/api/health") == {"ok": True, "runtime": "mahjong_agent_runtime"}
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        assert get_health(runtime_port, "/api/health") is None
        assert get_health(adapter_port, "/health") is None
        assert get_health(wx_port, "/health") == {"status": "ok"}
