"""Wait for a sidecar job to reach a terminal state, with a hard deadline.

The CLI submits work to the API sidecar and must never hang: it polls the
job status, stops on a terminal state, and gives up after ``timeout_s``.
The status fetcher is injected so this module has no HTTP dependency and
can be tested without a server.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, Unpack

ACTIVE_STATES = frozenset({"pending", "running"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})


class JobStatusError(Exception):
    """The job status could not be read (transport failure or bad response)."""


class JobNotFoundError(JobStatusError):
    """The server does not know the job id; waiting longer cannot help."""


@dataclass(frozen=True)
class JobWaitResult:
    """Outcome of waiting on a job.

    ``state`` is the job's terminal state, or ``"timeout"`` / ``"unknown"``.
    ``exit_code`` is what the CLI should exit with (0 only for ``completed``).
    """

    state: str
    exit_code: int
    detail: str
    job: dict[str, Any] | None = None


def _result_for_terminal(job: dict[str, Any]) -> JobWaitResult:
    state = str(job.get("state"))
    reported = job.get("exit_code")
    if state == "completed":
        return JobWaitResult(state, 0, "Job completed.", job)
    exit_code = int(reported) if isinstance(reported, int) and reported != 0 else 1
    message = job.get("error_message") or f"Job ended in state '{state}'."
    return JobWaitResult(state, exit_code, str(message), job)


class WaitHooks(TypedDict, total=False):
    """Optional callbacks for :func:`wait_for_job`.

    ``on_state`` is called on every state change; ``sleep`` (default
    :func:`time.sleep`) and ``clock`` (default :func:`time.monotonic`) are
    injectable so tests can run on a fake clock.
    """

    on_state: Callable[[str], None] | None
    sleep: Callable[[float], None]
    clock: Callable[[], float]


def wait_for_job(
    fetch_status: Callable[[], dict[str, Any]],
    timeout_s: float,
    poll_interval_s: float = 1.0,
    **hooks: Unpack[WaitHooks],
) -> JobWaitResult:
    """Poll ``fetch_status`` until the job is terminal, unknown, or the deadline passes.

    ``fetch_status`` returns the job JSON, raises :class:`JobNotFoundError`
    when the job does not exist, and :class:`JobStatusError` for transient
    failures (those are retried until the deadline). ``hooks`` are the
    optional ``on_state``, ``sleep`` and ``clock`` callbacks (see
    :class:`WaitHooks`).
    """
    on_state = hooks.get("on_state")
    sleep = hooks.get("sleep", time.sleep)
    clock = hooks.get("clock", time.monotonic)
    deadline = clock() + timeout_s
    last_state: str | None = None
    last_error: str | None = None

    while True:
        try:
            job = fetch_status()
        except JobNotFoundError as exc:
            return JobWaitResult("unknown", 1, f"Job not found: {exc}")
        except JobStatusError as exc:
            last_error = str(exc)
        else:
            last_error = None
            state = str(job.get("state"))
            if state != last_state and on_state is not None:
                on_state(state)
            last_state = state
            if state in TERMINAL_STATES:
                return _result_for_terminal(job)
            if state not in ACTIVE_STATES:
                return JobWaitResult("unknown", 1, f"Job reported an unknown state '{state}'.", job)

        remaining = deadline - clock()
        if remaining <= 0:
            detail = f"Timed out after {timeout_s:g}s; last state: {last_state or 'none'}."
            if last_error:
                detail += f" Last status error: {last_error}"
            return JobWaitResult("timeout", 1, detail)
        sleep(min(poll_interval_s, remaining))
