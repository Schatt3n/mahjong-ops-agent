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


def init_test_git_repo(root: Path) -> None:
    if not shutil.which("git"):
        pytest.skip("Git is required to exercise linked worktrees")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         "-c", "user.name=Launcher Test", "-c", "user.email=launcher@example.invalid",
         "commit", "-q", "--allow-empty", "-m", "fixture"],
        check=True,
    )


def add_test_worktree(repo: Path, target: Path) -> Path:
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(target), "HEAD"], check=True)
    return target


def make_test_wechat_store(path: Path, *, token_mode: int = 0o600) -> Path:
    path.mkdir(mode=0o700)
    (path / "registry.sqlite").touch(mode=0o600)
    token = path / "api-token"
    token.write_text("test-wechat-token", encoding="utf-8")
    token.chmod(token_mode)
    return path.resolve()


@pytest.fixture
def data_dir_worktrees(tmp_path, monkeypatch):
    from scripts import start_wechat_stack as launcher

    prior = tmp_path / "prior checkout"
    init_test_git_repo(prior)
    current = add_test_worktree(prior, tmp_path / "current checkout")
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setattr(launcher, "ROOT", current)
    monkeypatch.chdir(current)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.delenv("MAHJONG_WECHAT_DATA_DIR", raising=False)
    return launcher, prior, xdg / "agent-wechat"


def test_userdata_explicit_fresh_path_wins_over_saved_stores(data_dir_worktrees, monkeypatch):
    launcher, prior, default = data_dir_worktrees
    make_test_wechat_store(default)
    make_test_wechat_store(prior / "saved")
    (prior / ".env.wechat-stack.local").write_text("MAHJONG_WECHAT_DATA_DIR=saved\n", encoding="utf-8")
    monkeypatch.setenv("MAHJONG_WECHAT_DATA_DIR", "fresh-isolated-state")

    assert launcher.resolve_wechat_data_dir(6174) == (launcher.ROOT / "fresh-isolated-state").resolve()


def test_userdata_uses_saved_xdg_default_before_prior_checkout(data_dir_worktrees):
    launcher, prior, default = data_dir_worktrees
    saved_default = make_test_wechat_store(default)
    make_test_wechat_store(prior / "saved")
    (prior / ".env").write_text("MAHJONG_WECHAT_DATA_DIR=saved\n", encoding="utf-8")

    assert launcher.resolve_wechat_data_dir(6174) == saved_default


def test_userdata_resolves_prior_relative_path_for_the_same_endpoint(data_dir_worktrees):
    launcher, prior, default = data_dir_worktrees
    make_test_wechat_store(prior / "other state")
    saved = make_test_wechat_store(prior / "saved state")
    (prior / ".env").write_text(
        "MAHJONG_WECHAT_DATA_DIR='other state'\nMAHJONG_AGENT_WECHAT_BASE_URL=http://127.0.0.1:6174\n",
        encoding="utf-8",
    )
    (prior / ".env.wechat-stack.local").write_text(
        "MAHJONG_WECHAT_DATA_DIR='saved state'\nMAHJONG_AGENT_WECHAT_BASE_URL=http://127.0.0.1:16174/\n",
        encoding="utf-8",
    )

    assert launcher.resolve_wechat_data_dir(16174) == saved
    assert launcher.resolve_wechat_data_dir(6174) == default.resolve()
    assert not default.exists()


def test_userdata_deduplicates_resolved_paths_and_rejects_distinct_matches(data_dir_worktrees):
    launcher, prior, _ = data_dir_worktrees
    sibling = add_test_worktree(prior, prior.parent / "other checkout")
    saved = make_test_wechat_store(prior / "saved")
    (prior / ".env").write_text("MAHJONG_WECHAT_DATA_DIR=saved\n", encoding="utf-8")
    config = sibling / ".env.wechat-stack.local"
    config.write_text(f"MAHJONG_WECHAT_DATA_DIR='{saved}'\n", encoding="utf-8")

    assert launcher.resolve_wechat_data_dir(6174) == saved

    make_test_wechat_store(sibling / "other")
    config.write_text("MAHJONG_WECHAT_DATA_DIR=other\n", encoding="utf-8")

    with pytest.raises(launcher.LaunchError, match="MAHJONG_WECHAT_DATA_DIR"):
        launcher.resolve_wechat_data_dir(6174)


def test_userdata_ignores_undeclared_and_uninitialized_prior_directories(data_dir_worktrees):
    launcher, prior, default = data_dir_worktrees
    make_test_wechat_store(prior / "undeclared")
    make_test_wechat_store(prior / "public-token", token_mode=0o644)
    uninitialized = prior / "uninitialized"
    uninitialized.mkdir(mode=0o700)
    (uninitialized / "api-token").touch(mode=0o600)
    config = prior / ".env.wechat-stack.local"

    for path in ("public-token", "uninitialized"):
        config.write_text(f"MAHJONG_WECHAT_DATA_DIR={path}\n", encoding="utf-8")
        assert launcher.resolve_wechat_data_dir(6174) == default.resolve()
    assert not default.exists()


@pytest.fixture
def launcher_with_clean_model_env(monkeypatch):
    from scripts import start_wechat_stack as launcher

    monkeypatch.setattr(os, "environ", {
        key: value for key, value in os.environ.items()
        if not key.startswith(("MAHJONG_LLM_", "MAHJONG_DEEPSEEK_", "DEEPSEEK_"))
    })
    monkeypatch.setattr(launcher, "configure_model_in_browser", lambda **_: None)
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


def test_browser_setup_saves_custom_route_and_wire_format(tmp_path, monkeypatch, launcher_with_clean_model_env):
    launcher = launcher_with_clean_model_env
    local = tmp_path / launcher.MODEL_ENV_NAME
    settings = {
        launcher.MODEL_PROVIDER: "openai_compatible",
        launcher.MODEL_ENDPOINT: "http://127.0.0.1:18888/v1",
        launcher.MODEL_NAME: "test-local-model",
        launcher.MODEL_KEY: "synthetic-key",
        launcher.MODEL_FORMAT: "",
        launcher.MODEL_PROTOCOL: "responses",
    }

    def submit(*, initial, validate):
        assert launcher.MODEL_KEY not in initial
        validate(settings)
        return settings

    monkeypatch.setattr(launcher, "configure_model_in_browser", submit)
    launcher.require_runtime_model(local)

    assert local.stat().st_mode & 0o777 == 0o600
    assert launcher.read_dotenv_values(local) == settings
    config = launcher.OpenAICompatibleAgentClient.from_env().config
    assert config.response_format == ""
    assert config.api_protocol == "responses"


def test_incomplete_provider_change_never_reuses_key_at_new_endpoint(tmp_path, launcher_with_clean_model_env):
    launcher = launcher_with_clean_model_env
    local = tmp_path / launcher.MODEL_ENV_NAME
    local.write_text(
        "MAHJONG_LLM_PROVIDER=deepseek\nMAHJONG_LLM_MODEL=deepseek-flash\nMAHJONG_LLM_API_KEY=old-test-key\n",
        encoding="utf-8",
    )
    os.environ.update({launcher.MODEL_PROVIDER: "openai_compatible", launcher.MODEL_ENDPOINT: "https://other.invalid/v1"})

    launcher.require_runtime_model(local)

    config = launcher.OpenAICompatibleAgentClient.from_env().config
    assert config.api_key == "old-test-key"
    assert config.base_url == "https://api.deepseek.com"


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
    ("login_mode", "bootstrap", "stop_during_qr", "refresh_qr"),
    [
        ("online", False, False, False),
        ("saved-trust", False, False, False),
        ("qr", False, False, False),
        ("qr", True, False, False),
        ("qr", False, True, False),
        ("qr", False, False, True),
    ],
    ids=["online-stop-command", "saved-device-trust", "qr-login", "missing-runtime-tools", "stop-during-qr", "refresh-qr"],
)
def test_stack_launches_runtime_and_adapter_with_local_wechat(
    tmp_path, login_mode, bootstrap, stop_during_qr, refresh_qr,
):
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
    prior_checkout = None
    if login_mode == "saved-trust":
        init_test_git_repo(checkout)
        prior_checkout = add_test_worktree(checkout, tmp_path / "prior checkout")
    token_file = make_test_wechat_store((prior_checkout or tmp_path) / "hub") / "api-token"
    online = tmp_path / "wechat-online"
    qr_path_receipt = tmp_path / "qr-path"
    closed_windows = tmp_path / "closed-qr-windows"
    refresh_requested = tmp_path / "refresh-qr"
    login_cancelled = tmp_path / "login-cancelled"
    login_strategy = tmp_path / "login-strategy"
    saved_trust = token_file.parent / "test-saved-device-trust"
    if login_mode == "online":
        online.touch()
    if login_mode == "saved-trust":
        saved_trust.touch()

    wx = tmp_path / "wx"
    wx.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "if '--qr-output' not in sys.argv:\n"
        "    sys.exit('logged-in daemon should not require another wx command')\n"
        "assert sys.argv[sys.argv.index('--account') + 1] == 'test-account'\n"
        "strategy = sys.argv[sys.argv.index('--strategy') + 1]\n"
        "assert strategy == 'auto'\n"
        f"with Path({str(login_strategy)!r}).open('a') as log:\n"
        "    log.write(strategy + '\\n')\n"
        f"online = Path({str(online)!r})\n"
        f"if Path({str(saved_trust)!r}).is_file():\n"
        "    online.touch()\n"
        "    sys.exit(0)\n"
        "qr = Path(sys.argv[sys.argv.index('--qr-output') + 1])\n"
        "qr.write_bytes(b'\\x89PNG\\r\\n\\x1a\\nmock-qr')\n"
        f"refresh = Path({str(refresh_requested)!r})\n"
        "try:\n"
        "    for _ in range(100):\n"
        "        if refresh.exists():\n"
        "            qr.with_name('login-2.png').write_bytes(qr.read_bytes())\n"
        "            refresh.unlink()\n"
        "        if online.exists():\n"
        "            print(json.dumps({'state': 'succeeded'}))\n"
        "            sys.exit(0)\n"
        "        time.sleep(0.1)\n"
        "except KeyboardInterrupt:\n"
        f"    Path({str(login_cancelled)!r}).touch()\n"
        "    sys.exit(130)\n"
        "sys.exit('test did not simulate scanning the QR code')\n",
        encoding="utf-8",
    )
    wx.chmod(0o700)
    viewer = tmp_path / ("osascript" if sys.platform == "darwin" else "xdg-open")
    viewer.write_text(
        f"#!{sys.executable}\n"
        "import signal, sys\n"
        "from pathlib import Path\n"
        "qr = Path(sys.argv[-1])\n"
        "if sys.platform == 'darwin':\n"
        "    assert sys.argv[1:3] == ['-l', 'JavaScript']\n"
        "    assert Path(sys.argv[3]).name == 'wechat_login_qr.js'\n"
        "    def close(*_):\n"
        "        assert qr.is_file()\n"
        f"        with Path({str(closed_windows)!r}).open('a') as log:\n"
        "            log.write(qr.name + '\\n')\n"
        "        sys.exit(0)\n"
        "    signal.signal(signal.SIGTERM, close)\n"
        f"receipt = Path({str(qr_path_receipt)!r})\n"
        "pending_receipt = receipt.with_suffix('.tmp')\n"
        "pending_receipt.write_text(str(qr))\n"
        "pending_receipt.replace(receipt)\n"
        "if sys.platform == 'darwin':\n"
        "    while True:\n"
        "        signal.pause()\n",
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
        if prior_checkout is not None:
            (prior_checkout / ".env.wechat-stack.local").write_text(
                f"MAHJONG_WECHAT_DATA_DIR=hub\nMAHJONG_AGENT_WECHAT_BASE_URL=http://127.0.0.1:{wx_port}/\n",
                encoding="utf-8",
            )
            env.pop("MAHJONG_WECHAT_DATA_DIR")
            env.pop("MAHJONG_AGENT_WECHAT_TOKEN_FILE")
            env["PATH"] += os.pathsep + str(Path(shutil.which("git")).parent)
        if bootstrap:
            env["PATH"] = str(system_bin)
            env.pop("MAHJONG_WECHAT_WX_BIN")
        command = [sys.executable, str(checkout / "scripts" / "start_wechat_stack.py")]

        def stop_stack(expected: str):
            result = subprocess.run([*command, "--stop"], cwd=checkout, env=env, capture_output=True, text=True, timeout=25)
            assert result.returncode == 0 and expected in result.stdout, result.stdout + result.stderr

        log = tmp_path / "launcher.log"
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
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
                    if login_mode == "qr" and qr is None and qr_path_receipt.exists():
                        qr = Path(qr_path_receipt.read_text(encoding="utf-8"))
                        assert qr.read_bytes().startswith(b"\x89PNG")
                        assert get_health(adapter_port, "/health") is None
                        if stop_during_qr:
                            break
                        if refresh_qr:
                            refresh_requested.touch()
                        else:
                            online.touch()
                    if refresh_qr and qr is not None and not online.exists():
                        refreshed = Path(qr_path_receipt.read_text(encoding="utf-8"))
                        if refreshed != qr:
                            assert refreshed.name == "login-2.png"
                            if sys.platform == "darwin":
                                assert closed_windows.read_text(encoding="utf-8").splitlines() == ["login.png"]
                            online.touch()
                    adapter = get_health(adapter_port, "/health")
                    if adapter:
                        break
                    time.sleep(0.05)

                if stop_during_qr:
                    assert qr is not None, log.read_text(encoding="utf-8")
                    stop_stack("已停止本次启动的服务。")
                    process.wait(timeout=5)
                    assert login_cancelled.exists()
                    assert not qr.parent.exists()
                    assert not (checkout / "data" / "agent_runtime.sqlite3").exists()
                    if sys.platform == "darwin":
                        assert closed_windows.read_text(encoding="utf-8").splitlines() == ["login.png"]
                else:
                    assert adapter, log.read_text(encoding="utf-8")
                    assert adapter["account_id"] == "test-account"
                    assert adapter["send_channel_enabled"] is False
                    assert adapter["auto_send_reply"] is False
                    assert qr_path_receipt.exists() == (login_mode == "qr")
                    if bootstrap:
                        assert bootstrap_log.read_text(encoding="utf-8").splitlines() == ["node", "wx"]
                    if qr is not None:
                        assert not qr.parent.exists()
                    if sys.platform == "darwin":
                        expected_closed = ["login.png"] if login_mode == "qr" else []
                        if refresh_qr:
                            expected_closed.append("login-2.png")
                        assert closed_windows.exists() == (login_mode == "qr")
                        if expected_closed:
                            assert closed_windows.read_text(encoding="utf-8").splitlines() == expected_closed
                    assert get_health(runtime_port, "/api/health") == {"ok": True, "runtime": "mahjong_agent_runtime"}
                if login_mode == "online":
                    assert not login_strategy.exists()
                    stop_stack("已停止本次启动的服务。")
                    process.wait(timeout=5)
                    stop_stack("当前没有由一键入口管理的服务。")
                else:
                    assert login_strategy.read_text(encoding="utf-8").splitlines() == ["auto"]
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
