"""通过项目私有 Unix socket 请求启动器收尾，不从磁盘获取待发送信号的 PID。"""

from __future__ import annotations

import fcntl
from pathlib import Path
import signal
import socket
import stat
from threading import Event, Thread, get_ident
import time
import os


LOCK_PATH = Path("data/wechat-stack.lock")
SOCKET_PATH = Path("data/wechat-stack-control/stop.sock")


class StopControl:
    """由持有项目启动锁的进程发布，全部子进程收尾后才向停止请求回执。"""

    def __init__(self) -> None:
        self.listener: socket.socket | None = None
        self.worker: Thread | None = None
        self.closed = Event()
        self.clients: list[socket.socket] = []
        self.owner_thread = get_ident()

    def start(self) -> None:
        """在私有目录监听。相对地址让中文或深层 checkout 也能避开 sun_path 限长。"""
        directory = SOCKET_PATH.parent
        directory.mkdir(mode=0o700, exist_ok=True)
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise OSError(f"停止控制目录须由当前用户独占：{directory}")
        try:
            existing = SOCKET_PATH.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(existing.st_mode):
                raise OSError(f"停止控制路径已有其他文件：{SOCKET_PATH}")
            # 此时已持有排他锁，遗留的 socket 不可能属于另一启动器。
            SOCKET_PATH.unlink()

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(SOCKET_PATH))
            listener.listen(8)
            listener.settimeout(0.2)
            self.listener = listener.dup()
        self.worker = Thread(target=self._accept, daemon=True, name="wechat-stack-stop")
        self.worker.start()

    def _accept(self) -> None:
        """首次连接通知主线程，后续连接共享同一轮退出结果。"""
        with self.listener as listener:
            while not self.closed.is_set():
                try:
                    client, _ = listener.accept()
                except TimeoutError:
                    continue
                self.clients.append(client)
                if len(self.clients) == 1:
                    signal.pthread_kill(self.owner_thread, signal.SIGTERM)

    def finish(self, result: str) -> None:
        """停止接入并回传最终结果。断开的停止窗口不影响主进程收尾。"""
        if self.listener is None:
            return
        self.closed.set()
        if self.worker is not None and self.worker.ident is not None:
            self.worker.join()
        self.listener.close()
        SOCKET_PATH.unlink(missing_ok=True)
        for client in self.clients:
            with client:
                try:
                    client.settimeout(0.2)
                    client.sendall(result.encode("ascii") + b"\n")
                except OSError:
                    pass


def stack_is_managed() -> bool:
    """用启动锁区分活动启动器和旧 socket，不扫描端口或其他进程。"""
    try:
        with LOCK_PATH.open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except FileNotFoundError:
        return False
    except BlockingIOError:
        return True
    return False


def stop_running_stack() -> int:
    """请求活动启动器退出，只有收到子进程收尾回执才报告停止。"""
    deadline = time.monotonic() + 3
    try:
        while stack_is_managed():
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                try:
                    client.connect(str(SOCKET_PATH))
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)
                    continue
                print("正在停止棋牌室运营，请等待服务收尾。", flush=True)
                client.settimeout(185)
                with client.makefile("rb") as response:
                    result = response.readline(16)
            if result == b"stopped\n":
                print("已停止本次启动的服务。", flush=True)
                return 0
            if result == b"forced\n":
                print("已停止本次启动的服务；超时组件已强制结束，详情见启动窗口。", flush=True)
                return 0
            break
        else:
            print("当前没有由一键入口管理的服务。", flush=True)
            return 0
    except OSError:
        pass
    print("尚未确认全部服务退出，请查看启动窗口。", flush=True)
    return 1
