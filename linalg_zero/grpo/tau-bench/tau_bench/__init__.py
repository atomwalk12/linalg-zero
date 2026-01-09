# Copyright Sierra

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tau_bench.agents.base import Agent as Agent
    from tau_bench.envs.base import Env as Env

__all__ = ["Agent", "Env"]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from tau_bench.agents.base import Agent as _Agent

        return _Agent
    if name == "Env":
        from tau_bench.envs.base import Env as _Env

        return _Env
    raise AttributeError(name)
