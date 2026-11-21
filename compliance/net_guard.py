"""离线运行防护。

赛题要求在评测环境断开外网，禁止调用外部模型 API。这里在进程启动时替换
socket 的出站入口：任何尝试建立网络连接的行为都会立即抛错并记录，避免
依赖库在后台偷偷联网（例如自动下载权重、遥测）。
"""
from __future__ import annotations

import socket
import threading
from typing import Any

_attempts: list[str] = []
_lock = threading.Lock()


class OfflineViolation(RuntimeError):
    """代码尝试访问网络。"""


def _record(target: str) -> None:
    with _lock:
        _attempts.append(target)


def _blocked_connect(self: socket.socket, address: Any) -> Any:
    _record(f"connect {address!r}")
    raise OfflineViolation(
        f"离线评测禁止网络连接，已拦截 connect({address!r})"
    )


def _blocked_connect_ex(self: socket.socket, address: Any) -> int:
    _record(f"connect_ex {address!r}")
    raise OfflineViolation(
        f"离线评测禁止网络连接，已拦截 connect_ex({address!r})"
    )


def _blocked_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    _record(f"create_connection {address!r}")
    raise OfflineViolation(
        f"离线评测禁止网络连接，已拦截 create_connection({address!r})"
    )


def _blocked_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    _record(f"getaddrinfo {host!r}")
    raise OfflineViolation(
        f"离线评测禁止域名解析，已拦截 getaddrinfo({host!r})"
    )


def enable_offline_guard() -> None:
    """幂等地关闭进程内的出站网络。"""
    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _blocked_getaddrinfo  # type: ignore[assignment]


def attempted_connections() -> list[str]:
    with _lock:
        return list(_attempts)
