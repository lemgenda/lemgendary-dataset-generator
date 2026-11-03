"""
MetaTrader 5 session handling for the download pipeline.

Connects to and disconnects from the terminal, and resolves logical pair
names (e.g. ``USOIL``) to the symbol names the connected broker uses.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from . import bridge as mt5_bridge
from .pipeline_config import ALTERNATIVE_KEYWORDS

__all__ = [
    "MT5Credentials",
    "connect_mt5",
    "disconnect_mt5",
    "mt5_call",
    "remember_symbol",
    "resolve_symbol",
]

logger = logging.getLogger(__name__)

_MT5_ERRORS = (RuntimeError, AttributeError, OSError)

_SYMBOL_CACHE: dict[str, str] = {}  # pair -> resolved symbol name


@dataclass(frozen=True)
class MT5Credentials:
    """Optional MT5 login details; missing fields fall back to ``MT5_*`` env variables."""

    login: int | None = None
    password: str | None = None
    server: str | None = None
    api_key: str | None = None


def mt5_call(method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``method_name`` on the MetaTrader5 module through the bridge."""
    return mt5_bridge.call(method_name, *args, **kwargs)


def _report_account() -> bool:
    """Print the connected account and return True, or return False without account info."""
    info = mt5_call("account_info")
    if not info:
        return False
    print(
        f" [MT5] Connected -> Account: {info.login} | Server: {info.server} "
        f"| Balance: {info.balance} {info.currency}"
    )
    return True


def _attach_running_terminal() -> bool:
    """Try to attach to an already running, logged-in terminal."""
    try:
        if mt5_call("initialize"):
            return _report_account()
    except _MT5_ERRORS as e:
        logger.warning("MT5 attach to running terminal failed: %s", e)
        print(f" [MT5] Failed to attach to terminal: {e}")
    return False


def _env_login() -> int | None:
    if "MT5_LOGIN" not in os.environ:
        return None
    try:
        return int(os.environ["MT5_LOGIN"])
    except ValueError as err:
        print(f" [MT5] Warning: Invalid MT5_LOGIN integer format: {err}")
        return None


def _with_env_defaults(creds: MT5Credentials) -> MT5Credentials:
    """Fill the missing credential fields from the ``MT5_*`` environment variables."""
    return MT5Credentials(
        login=creds.login if creds.login is not None else _env_login(),
        password=creds.password if creds.password is not None else os.environ.get("MT5_PASSWORD"),
        server=creds.server if creds.server is not None else os.environ.get("MT5_SERVER"),
        api_key=creds.api_key if creds.api_key is not None else os.environ.get("MT5_API_KEY"),
    )


def _login_with(creds: MT5Credentials) -> bool:
    """Initialize MT5 with explicit login details; raises RuntimeError on failure."""
    init_kwargs = {"login": creds.login, "password": creds.password, "server": creds.server}
    try:
        if mt5_call("initialize", **init_kwargs):
            if _report_account():
                return True
            raise RuntimeError("[MT5] Connected but no account info.")
        err = mt5_call("last_error")
        raise RuntimeError(f"[MT5] Initialize failed: {err}")
    except _MT5_ERRORS as e:
        raise RuntimeError(f"[MT5] Initialize exception: {e}") from e


def connect_mt5(login=None, password=None, server=None, api_key=None):
    """Connect to MetaTrader 5, logging in when credentials are given; returns True on success."""
    if _attach_running_terminal():
        return True

    creds = _with_env_defaults(
        MT5Credentials(login=login, password=password, server=server, api_key=api_key)
    )
    if creds.login and creds.password and creds.server:
        return _login_with(creds)
    raise RuntimeError(
        "[MT5] Could not connect to MetaTrader 5.\nPlease ensure:\n  1. MetaTrader 5 is running"
        " and logged in\n  2. If using a demo account, it is active\n  3. If you need explicit"
        " login, set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER environment variables"
    )


def disconnect_mt5():
    """Shut down the MetaTrader 5 connection."""
    try:
        mt5_call("shutdown")
        print(" [MT5] Disconnected.")
    except _MT5_ERRORS as err:
        logger.warning("MT5 shutdown failed: %s", err)
        print(f" [MT5] Warning during shutdown: {err}")


# ─── Symbol Resolution (dynamic, caches results) ────────────────────────


def _usable_alternatives(candidates):
    return [alt for alt in candidates if len(alt) >= 3]


def _is_prefixed_name(name_upper: str, alt: str) -> bool:
    """True when ``name_upper`` is ``alt`` or ``alt`` followed by a ``.``, ``_`` or ``-`` suffix."""
    return name_upper.startswith(alt) and (
        len(name_upper) == len(alt) or name_upper[len(alt)] in ('.', '_', '-')
    )


def _find_matching_symbol(all_symbols, candidates):
    for sym in all_symbols:
        if sym.name.upper() in candidates:
            return sym.name

    for alt in _usable_alternatives(candidates):
        for sym in all_symbols:
            if _is_prefixed_name(sym.name.upper(), alt):
                return sym.name

    for alt in _usable_alternatives(candidates):
        for sym in all_symbols:
            if alt in sym.name.upper():
                return sym.name
    return None


def remember_symbol(pair: str, mt5_symbol: str) -> None:
    """Cache ``mt5_symbol`` as the MT5 symbol that serves ``pair``."""
    _SYMBOL_CACHE[pair] = mt5_symbol


def resolve_symbol(pair: str) -> str | None:
    """Find the actual MT5 symbol name for a given logical pair."""
    if pair in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[pair]

    try:
        all_symbols = mt5_call("symbols_get")
    except _MT5_ERRORS as err:
        logger.warning("MT5 symbols_get failed while resolving %s: %s", pair, err)
        return None

    if not all_symbols:
        return None

    pair_upper = pair.upper()
    candidates = [pair_upper] + [alt.upper() for alt in ALTERNATIVE_KEYWORDS.get(pair, [])]
    matched = _find_matching_symbol(all_symbols, candidates)
    if matched:
        remember_symbol(pair, matched)
        return matched
    return None
