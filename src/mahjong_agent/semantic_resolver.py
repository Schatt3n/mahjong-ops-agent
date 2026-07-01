from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .action_arguments_contract import validate_action_arguments_contract
from .profile_observation_contract import validate_profile_observation_contract
from .workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    GameRequirement,
    ProposedAction,
    RiskLevel,
    SemanticResolution,
    SlotSource,
    SlotValue,
    UserIntent,
)


DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "semantic_resolution.md"


class SemanticLLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str | dict[str, Any]:
        ...


@dataclass(slots=True)
class SemanticResolverConfig:
    prompt_path: Path = DEFAULT_PROMPT_PATH
    timeout_seconds: float = 8.0
    include_prompt_in_raw_response: bool = True
    allow_json_fragment_extraction: bool = False


class SemanticResolver:
    """LLM semantic resolver for the controlled workflow.

    It builds the prompt, calls an injected LLM client, and converts model JSON
    into SemanticResolution. It does not call tools or mutate application state.
    """

    def __init__(
        self,
        llm_client: SemanticLLMClient,
        config: SemanticResolverConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SemanticResolverConfig()
        self._prompt_cache: str | None = None

    def resolve(self, context: ConversationContext) -> SemanticResolution:
        messages = self.build_messages(context)
        trace_id = context.current_message.trace_id
        try:
            raw_output = self.llm_client.complete(
                messages,
                trace_id=trace_id,
                timeout_seconds=self.config.timeout_seconds,
            )
        except TimeoutError as exc:
            return self._failure_resolution(
                reason=f"LLM semantic resolver timeout: {exc}",
                raw_response={
                    "error": f"{type(exc).__name__}: {exc}",
                    "llm_contract": self._llm_contract_audit(
                        messages=messages,
                        accepted=False,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                },
                prompt_messages=messages,
            )
        except Exception as exc:
            return self._failure_resolution(
                reason=f"LLM semantic resolver error: {type(exc).__name__}: {exc}",
                raw_response={
                    "error": f"{type(exc).__name__}: {exc}",
                    "llm_contract": self._llm_contract_audit(
                        messages=messages,
                        accepted=False,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                },
                prompt_messages=messages,
            )

        raw, parse_error = self._parse_raw_output(raw_output)
        if parse_error:
            return self._failure_resolution(
                reason=parse_error,
                raw_response={
                    "raw_output": raw_output,
                    "parse_error": parse_error,
                    "llm_contract": self._llm_contract_audit(
                        messages=messages,
                        accepted=False,
                        raw_output=raw_output,
                        parse_error=parse_error,
                    ),
                },
                prompt_messages=messages,
            )
        contract_errors = _validate_semantic_contract(raw)
        if contract_errors:
            reason = "LLM semantic resolver contract invalid: " + "; ".join(contract_errors)
            return self._failure_resolution(
                reason=reason,
                raw_response={
                    "raw_output": raw_output,
                    "contract_errors": contract_errors,
                    "llm_contract": self._llm_contract_audit(
                        messages=messages,
                        accepted=False,
                        raw_output=raw_output,
                        contract_errors=contract_errors,
                    ),
                },
                prompt_messages=messages,
            )
        return self._resolution_from_raw(raw, prompt_messages=messages)

    def build_messages(self, context: ConversationContext) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._prompt_text()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "semantic_resolution_contract_v1",
                        "context": context.to_prompt_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]

    def _prompt_text(self) -> str:
        if self._prompt_cache is None:
            self._prompt_cache = self.config.prompt_path.read_text(encoding="utf-8")
        return self._prompt_cache

    def _parse_raw_output(self, raw_output: str | dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        if isinstance(raw_output, dict):
            return raw_output, None
        text = str(raw_output or "").strip()
        if not text:
            return {}, "LLM semantic resolver returned empty output."
        try:
            raw = json.loads(text)
            if not isinstance(raw, dict):
                return {}, "LLM semantic resolver JSON root is not an object."
            return raw, None
        except json.JSONDecodeError:
            if not self.config.allow_json_fragment_extraction:
                return {}, "LLM semantic resolver output must be a single JSON object with no surrounding text."
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}, "LLM semantic resolver returned no JSON object."
            try:
                raw = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                return {}, f"LLM semantic resolver returned invalid JSON: {exc}"
            if not isinstance(raw, dict):
                return {}, "LLM semantic resolver JSON fragment is not an object."
            return raw, None

    def _resolution_from_raw(
        self,
        raw: dict[str, Any],
        *,
        prompt_messages: list[dict[str, str]],
    ) -> SemanticResolution:
        confidence = _coerce_confidence(raw.get("confidence"))
        intent = _intent_from_raw(raw.get("intent"))
        action_name = _action_from_raw(raw.get("proposed_action"), intent=intent)
        risk_level = _risk_for_action(action_name, bool(raw.get("needs_human_review")))
        reasoning_summary = _optional_str(raw.get("reasoning_summary")) or _optional_str(raw.get("reason")) or ""
        proposed_action = ProposedAction(
            name=action_name,
            source=ActionSource.LLM,
            confidence=confidence,
            reason=reasoning_summary or "LLM semantic resolution",
            arguments=dict(raw.get("action_arguments") or {}) if isinstance(raw.get("action_arguments"), dict) else {},
            risk_level=risk_level,
        )
        game_requirement = GameRequirement()
        slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else {}
        for name, slot_raw in slots.items():
            game_requirement.set_slot(_slot_from_raw(str(name), slot_raw, default_confidence=confidence))

        raw_response = {
            "model_output": raw,
            "schema": "semantic_resolution_contract_v1",
            "llm_contract": self._llm_contract_audit(
                messages=prompt_messages,
                accepted=True,
                raw_output=raw,
            ),
        }
        if self.config.include_prompt_in_raw_response:
            raw_response["prompt_messages"] = list(prompt_messages)

        return SemanticResolution(
            intent=intent,
            proposed_action=proposed_action,
            game_requirement=game_requirement,
            needs_human_review=bool(raw.get("needs_human_review")) or risk_level == RiskLevel.HIGH,
            reasoning_summary=reasoning_summary,
            raw_response=raw_response,
        )

    def _llm_contract_audit(
        self,
        *,
        messages: list[dict[str, str]],
        accepted: bool,
        raw_output: Any | None = None,
        parse_error: str | None = None,
        contract_errors: list[str] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        audit: dict[str, Any] = {
            "schema": "semantic_resolution_contract_v1",
            "attempted": True,
            "accepted": accepted,
            "strict_json": not self.config.allow_json_fragment_extraction,
        }
        if raw_output is not None:
            audit["raw_output"] = raw_output
        if parse_error:
            audit["parse_error"] = parse_error
        if contract_errors:
            audit["contract_errors"] = list(contract_errors)
        if error:
            audit["error"] = error
        if self.config.include_prompt_in_raw_response:
            audit["prompt_messages"] = list(messages)
        return audit

    def _failure_resolution(
        self,
        *,
        reason: str,
        raw_response: dict[str, Any],
        prompt_messages: list[dict[str, str]],
    ) -> SemanticResolution:
        if self.config.include_prompt_in_raw_response:
            raw_response = {**raw_response, "prompt_messages": list(prompt_messages)}
        return SemanticResolution(
            intent=UserIntent.UNKNOWN,
            proposed_action=ProposedAction(
                name=ActionName.HUMAN_REVIEW,
                source=ActionSource.LLM,
                confidence=0.0,
                reason=reason,
                risk_level=RiskLevel.HIGH,
            ),
            needs_human_review=True,
            reasoning_summary=reason,
            raw_response=raw_response,
        )


SEMANTIC_REQUIRED_FIELDS: tuple[str, ...] = (
    "intent",
    "proposed_action",
    "confidence",
    "reasoning_summary",
    "slots",
)

SEMANTIC_ALLOWED_INTENTS = frozenset(item.value for item in UserIntent)
SEMANTIC_ALLOWED_ACTIONS = frozenset(item.value for item in ActionName)
SEMANTIC_ALLOWED_SLOT_SOURCES = frozenset(item.value for item in SlotSource)
SEMANTIC_SLOT_REQUIRED_FIELDS: tuple[str, ...] = (
    "value",
    "source",
    "confidence",
    "confirmed",
    "needs_confirmation",
)


def _validate_semantic_contract(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in SEMANTIC_REQUIRED_FIELDS:
        if field not in raw:
            errors.append(f"missing required field {field!r}")

    intent = raw.get("intent")
    if "intent" in raw and str(intent or "").strip() not in SEMANTIC_ALLOWED_INTENTS:
        errors.append(f"invalid intent {intent!r}")

    proposed_action = raw.get("proposed_action")
    if "proposed_action" in raw and str(proposed_action or "").strip() not in SEMANTIC_ALLOWED_ACTIONS:
        errors.append(f"invalid proposed_action {proposed_action!r}")

    confidence = raw.get("confidence")
    if "confidence" in raw:
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            errors.append(f"invalid confidence {confidence!r}")
        else:
            if numeric_confidence < 0 or numeric_confidence > 1:
                errors.append(f"confidence out of range {confidence!r}")

    reasoning_summary = raw.get("reasoning_summary")
    if "reasoning_summary" in raw and not _optional_str(reasoning_summary):
        errors.append("reasoning_summary must be a non-empty string")

    slots = raw.get("slots")
    if "slots" in raw:
        if not isinstance(slots, dict):
            errors.append("slots must be an object")
        else:
            errors.extend(_validate_slot_contracts(slots))

    if "needs_human_review" in raw and not isinstance(raw.get("needs_human_review"), bool):
        errors.append("needs_human_review must be a boolean when provided")

    if "action_arguments" in raw:
        action_for_arguments = _action_from_raw(
            raw.get("proposed_action"),
            intent=_intent_from_raw(raw.get("intent")),
        )
        errors.extend(validate_action_arguments_contract(action_for_arguments, raw.get("action_arguments")))

    profile_observations = raw.get("profile_observations")
    if "profile_observations" in raw:
        if not isinstance(profile_observations, list):
            errors.append("profile_observations must be an array when provided")
        else:
            for index, observation in enumerate(profile_observations):
                errors.extend(validate_profile_observation_contract(observation, index=index))

    return errors


def _validate_slot_contracts(slots: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for raw_name, raw_slot in slots.items():
        name = str(raw_name)
        if not isinstance(raw_slot, dict):
            errors.append(f"slot {name!r} must be an object")
            continue
        for field in SEMANTIC_SLOT_REQUIRED_FIELDS:
            if field not in raw_slot:
                errors.append(f"slot {name!r} missing required field {field!r}")
        if "value" in raw_slot and raw_slot.get("value") in (None, "", "unknown"):
            errors.append(f"slot {name!r} value must be non-empty")
        source = raw_slot.get("source")
        if "source" in raw_slot and str(source or "").strip() not in SEMANTIC_ALLOWED_SLOT_SOURCES:
            errors.append(f"slot {name!r} invalid source {source!r}")
        confidence = raw_slot.get("confidence")
        if "confidence" in raw_slot:
            try:
                numeric_confidence = float(confidence)
            except (TypeError, ValueError):
                errors.append(f"slot {name!r} invalid confidence {confidence!r}")
            else:
                if numeric_confidence < 0 or numeric_confidence > 1:
                    errors.append(f"slot {name!r} confidence out of range {confidence!r}")
        if "confirmed" in raw_slot and not isinstance(raw_slot.get("confirmed"), bool):
            errors.append(f"slot {name!r} confirmed must be a boolean")
        if "needs_confirmation" in raw_slot and not isinstance(raw_slot.get("needs_confirmation"), bool):
            errors.append(f"slot {name!r} needs_confirmation must be a boolean")
        if isinstance(raw_slot.get("confirmed"), bool) and isinstance(raw_slot.get("needs_confirmation"), bool):
            if raw_slot["confirmed"] == raw_slot["needs_confirmation"]:
                errors.append(f"slot {name!r} confirmed and needs_confirmation are inconsistent")
        if "metadata" in raw_slot and not isinstance(raw_slot.get("metadata"), dict):
            errors.append(f"slot {name!r} metadata must be an object when provided")
    return errors


def _slot_from_raw(name: str, raw: Any, *, default_confidence: float) -> SlotValue:
    if isinstance(raw, dict):
        value = raw.get("value")
        source = _slot_source_from_raw(raw.get("source"))
        confidence = _coerce_confidence(raw.get("confidence"), default=default_confidence)
        confirmed = raw.get("confirmed")
        needs_confirmation = raw.get("needs_confirmation")
        evidence = _optional_str(raw.get("evidence"))
        metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
    else:
        value = raw
        source = SlotSource.INFERRED
        confidence = default_confidence
        confirmed = None
        needs_confirmation = None
        evidence = None
        metadata = {}

    if confirmed is None:
        confirmed = source in {SlotSource.EXPLICIT, SlotSource.CONTEXT, SlotSource.TOOL} and confidence >= 0.75
    confirmed = bool(confirmed)
    if needs_confirmation is None:
        needs_confirmation = not confirmed
    return SlotValue(
        name=name,
        value=value,
        source=source,
        confidence=confidence,
        confirmed=confirmed,
        needs_confirmation=bool(needs_confirmation),
        evidence=evidence,
        metadata=metadata,
    )


def _slot_source_from_raw(value: Any) -> SlotSource:
    try:
        return SlotSource(str(value or SlotSource.UNKNOWN.value))
    except ValueError:
        return SlotSource.UNKNOWN


def _intent_from_raw(value: Any) -> UserIntent:
    intent = str(value or "").strip().lower()
    aliases = {
        "uncertain": UserIntent.UNKNOWN,
        "find_players": UserIntent.FIND_PLAYERS,
        "create_game": UserIntent.FIND_PLAYERS,
        "inquire_existing_games": UserIntent.INQUIRE_EXISTING_GAME,
        "search_existing_games": UserIntent.INQUIRE_EXISTING_GAME,
        "cancel_or_full": UserIntent.CANCEL_GAME,
        "cancel": UserIntent.CANCEL_GAME,
        "join": UserIntent.JOIN_GAME,
        "candidate_accept": UserIntent.CANDIDATE_REPLY,
        "no_reply": UserIntent.IRRELEVANT,
        "ignore": UserIntent.IRRELEVANT,
    }
    if intent in aliases:
        return aliases[intent]
    try:
        return UserIntent(intent)
    except ValueError:
        return UserIntent.UNKNOWN


def _action_from_raw(value: Any, *, intent: UserIntent) -> ActionName:
    action = str(value or "").strip().lower()
    aliases = {
        "search_existing": ActionName.SEARCH_EXISTING_GAMES,
        "search_current_open_games": ActionName.SEARCH_EXISTING_GAMES,
        "find_existing_game": ActionName.SEARCH_EXISTING_GAMES,
        "ask_create": ActionName.ASK_CREATE_CONFIRMATION,
        "create_new_game": ActionName.CREATE_GAME,
        "find_players": ActionName.CREATE_GAME,
        "clarify": ActionName.ASK_CLARIFICATION,
        "ask_followup": ActionName.ASK_CLARIFICATION,
        "manual_review": ActionName.HUMAN_REVIEW,
        "human": ActionName.HUMAN_REVIEW,
        "silent": ActionName.IGNORE,
        "no_reply": ActionName.IGNORE,
    }
    if action in aliases:
        return aliases[action]
    try:
        return ActionName(action)
    except ValueError:
        if intent == UserIntent.INQUIRE_EXISTING_GAME:
            return ActionName.SEARCH_EXISTING_GAMES
        if intent == UserIntent.FIND_PLAYERS:
            return ActionName.CREATE_GAME
        if intent == UserIntent.JOIN_GAME or intent == UserIntent.CANDIDATE_REPLY:
            return ActionName.JOIN_GAME
        if intent == UserIntent.CANCEL_GAME:
            return ActionName.CANCEL_GAME
        if intent == UserIntent.IRRELEVANT:
            return ActionName.IGNORE
        return ActionName.UNKNOWN


def _risk_for_action(action_name: ActionName, needs_human_review: bool) -> RiskLevel:
    if needs_human_review or action_name == ActionName.HUMAN_REVIEW:
        return RiskLevel.HIGH
    if action_name in {ActionName.CANCEL_GAME, ActionName.CLOSE_GAME, ActionName.CREATE_GAME, ActionName.QUEUE_INVITES}:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(default if value is None else value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None
