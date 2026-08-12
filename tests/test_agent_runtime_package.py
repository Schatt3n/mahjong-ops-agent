from __future__ import annotations

from pathlib import Path

from mahjong_agent_runtime import AgentRuntime, ContextSummaryManager, ContextSummaryPolicy, ToolGateway


ROOT = Path(__file__).resolve().parents[1]


def test_stable_runtime_package_exposes_versionless_api() -> None:
    import mahjong_agent_runtime

    assert mahjong_agent_runtime.AgentRuntime is AgentRuntime
    assert mahjong_agent_runtime.ContextSummaryManager is ContextSummaryManager
    assert mahjong_agent_runtime.ContextSummaryPolicy is ContextSummaryPolicy
    assert mahjong_agent_runtime.ToolGateway is ToolGateway
    assert not hasattr(mahjong_agent_runtime, "AgentRuntimeV3")
    assert not hasattr(mahjong_agent_runtime, "ToolGatewayV3")


def test_stable_runtime_package_does_not_import_compatibility_package() -> None:
    for path in (ROOT / "src" / "mahjong_agent_runtime").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mahjong_agent_v3" not in text, f"{path} should not import compatibility package"


def test_historical_v3_package_has_been_removed_from_main_repo() -> None:
    assert not (ROOT / "src" / "mahjong_agent_v3").exists()
    assert not (ROOT / "scripts" / "run_agent_v3_app.py").exists()
    assert not (ROOT / "scripts" / "run_agent_runtime_v3_eval.py").exists()
    assert not (ROOT / "scripts" / "verify_agent_runtime_v3_boundary.py").exists()
    assert not (ROOT / "docs" / "agent_runtime_v3.md").exists()


def test_v2_compatibility_modules_forward_to_canonical_implementations() -> None:
    from mahjong_agent_runtime.context import AgentContextBuilder as CompatContextBuilder
    from mahjong_agent_runtime.domains.context_builders import AgentContextBuilder
    from mahjong_agent_runtime.domains.game_domain import apply_game_lifecycle
    from mahjong_agent_runtime.domains.tools import ToolGateway
    from mahjong_agent_runtime.lifecycle import ContextLifecycleManager as CompatContextLifecycleManager
    from mahjong_agent_runtime.loop import AgentLoop as CompatAgentLoop
    from mahjong_agent_runtime.processing import ActionProcessor as CompatActionProcessor
    from mahjong_agent_runtime.services import ActionProcessor, AgentLoop, ContextLifecycleManager
    from mahjong_agent_runtime.sqlite_store import SQLiteAgentStore as CompatSQLiteAgentStore
    from mahjong_agent_runtime.store import (
        InMemoryAgentStore as CompatInMemoryAgentStore,
        apply_game_lifecycle as compat_apply_game_lifecycle,
    )
    from mahjong_agent_runtime.stores.memory import InMemoryAgentStore
    from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore
    from mahjong_agent_runtime.tools import ToolGateway as CompatToolGateway

    assert CompatContextBuilder is AgentContextBuilder
    assert CompatAgentLoop is AgentLoop
    assert CompatContextLifecycleManager is ContextLifecycleManager
    assert CompatActionProcessor is ActionProcessor
    assert CompatToolGateway is ToolGateway
    assert CompatInMemoryAgentStore is InMemoryAgentStore
    assert CompatSQLiteAgentStore is SQLiteAgentStore
    assert compat_apply_game_lifecycle is apply_game_lifecycle


def test_store_compatibility_facade_has_an_explicit_public_surface() -> None:
    from mahjong_agent_runtime import store

    assert len(store.__all__) == len(set(store.__all__))
    assert all(hasattr(store, name) for name in store.__all__)
    assert "InMemoryAgentStore" in store.__all__
    assert "normalize_requirement" in store.__all__
    assert "_release_game_participants" not in store.__all__
    assert "_rewrite_contact_names" not in store.__all__
