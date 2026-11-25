"""用已审核历史前缀里的表达线索丰富合成顾客，保留匿名来源。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, replace
import random
import re
from typing import Any, Literal

from .venue_actors import Persona
from .venue_history_seed import DEFAULT_HISTORY_CASE_ID, DEFAULT_HISTORY_DATASET, load_historical_group_seed
from .venue_private_history_seed import DEFAULT_PRIVATE_HISTORY_DATASET, load_historical_private_seed


_NOV22 = DEFAULT_HISTORY_DATASET.with_name("real_group_chat_20261122.jsonl")
_REVIEWED_GROUP_PREFIXES = (
    (DEFAULT_HISTORY_DATASET, DEFAULT_HISTORY_CASE_ID, "m04"),
    (_NOV22, "real_group_smoke_alternative_20261122_001", "m02"),
    (_NOV22, "real_group_reported_headcount_20261122_001", "m01"),
)
_CLOCK_NUMBER = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3])[:：][0-5]\d(?!\d)")
_CLOCK_SPOKEN = re.compile(r"(?:[零〇一二三四五六七八九十两\d]{1,3})点(?:钟|半|\d{1,2}分)?")
_CLOCK_PERIOD = re.compile(r"凌晨|早上|上午|中午|下午|傍晚|晚上|今晚|今早|明晚")
_CLAUSE_BREAK = re.compile(r"[，,；;。！？!?\n]")
_SMOKE_QUESTION = re.compile(r"(?:无烟|有烟|抽烟|吸烟)\s*[吗么呢?？]|(?:有烟|无烟)\s*还是\s*(?:有烟|无烟)")
_COORDINATION_QUESTION = re.compile(r"(?:有人|空位|够人|有空|有局|有桌|有位)(?:来|打)?\s*[吗么呢?？]|几点(?:开|到|来)")


@dataclass(frozen=True, slots=True)
class VenueRoleTrait:
    """一位匿名发言者在一个已审核前缀中可核对的表达现象。"""

    dataset: str
    case_id: str
    cutoff: str
    channel: Literal["group", "private"]
    actor_alias: str
    message_refs: tuple[str, ...]
    message_shape: Literal["brief", "multi_clause"] | None
    punctuation_style: str | None
    time_style: str | None
    question_topics: tuple[str, ...]
    quoted_delays_seconds: tuple[int, ...]
    version: int = 1

    @property
    def unknowns(self) -> tuple[str, ...]:
        """没有观察证据的项保持未知，引用间隔不能证明对方何时看到。"""
        absent = ("message_shape", "punctuation_style", "time_style", "question_topics", "quoted_delays_seconds")
        return ("long_term_preferences", "reading_time", "commitment", "relationships", *(
            name for name in absent if not getattr(self, name)
        ))


@dataclass(frozen=True, slots=True)
class VenueRoleAttribution:
    """仅供仿真本地审计的合成角色来源，与角色和店家输入独立保存。"""

    actor_id: str
    source: VenueRoleTrait
    speech_dimensions_changed: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """输出槽位来源和未证实项，供本地运行报告追溯。"""
        record = asdict(self)
        record["source"]["unknowns"] = self.source.unknowns
        return record


@dataclass(frozen=True, slots=True)
class VenueRoleBlend:
    """普通合成人物及独立的来源记录，模型只读取人物。"""

    people: tuple[Persona, ...]
    attributions: tuple[VenueRoleAttribution, ...]


def load_reviewed_venue_role_traits() -> tuple[VenueRoleTrait, ...]:
    """复用前缀加载器，只从审核切点前的顾客本人消息取特征。"""
    observed = []
    root = DEFAULT_HISTORY_DATASET.parents[2]
    for path, case_id, cutoff in _REVIEWED_GROUP_PREFIXES:
        history = load_historical_group_seed(path, case_id=case_id, cutoff=cutoff)
        offsets = {message.message_alias: message.offset_seconds for message in history.prefix}
        speakers = defaultdict(list)
        for message in history.prefix:
            text = message.text.strip()
            if not text or (text.startswith("[") and text.endswith("]")):
                continue
            interval = (message.offset_seconds - offsets[message.quoted_message_alias]
                        if message.quoted_message_alias is not None else None)
            speakers[message.actor_alias].append((message.message_alias, text, interval))
        observed.extend(_observed_trait(
            dataset=path.relative_to(root).as_posix(), case_id=case_id, cutoff=cutoff,
            channel="group", actor_alias=actor, samples=samples,
        ) for actor, samples in speakers.items())

    private = load_historical_private_seed()
    observed.append(_observed_trait(
        dataset=DEFAULT_PRIVATE_HISTORY_DATASET.relative_to(root).as_posix(), case_id=private.case_id,
        cutoff=str(private.cutoff_turn), channel="private", actor_alias="customer",
        samples=[(message.source_message_id, message.text.strip(), None)
                 for message in private.prefix if message.role == "customer"],
    ))
    return tuple(observed)


def _observed_trait(*, dataset: str, case_id: str, cutoff: str, channel: Literal["group", "private"],
                    actor_alias: str, samples: Sequence[tuple[str, str, int | None]]) -> VenueRoleTrait:
    """只描述可复核的长度、标点、时刻写法、问句题材和显式引用间隔。"""
    texts = tuple(text for _, text, _ in samples)
    clause_counts = [sum(bool(clause.strip()) for clause in _CLAUSE_BREAK.split(text)) for text in texts]
    shape = None
    if clause_counts and min(clause_counts) > 1:
        shape = "multi_clause"
    elif texts and max(clause_counts) == 1 and all(len("".join(text.split())) <= 16 for text in texts):
        shape = "brief"
    punctuation = "少用标点" if texts and not any(_CLAUSE_BREAK.search(text) for text in texts) else None
    if shape == "multi_clause" and all(re.search(r"[，,]", text) for text in texts):
        punctuation = "逗号串句"
    clocks = {style for text in texts for matches, style in (
        (_CLOCK_NUMBER.search(text), "数字时刻"),
        (_CLOCK_SPOKEN.search(text), "口语时段" if _CLOCK_PERIOD.search(text) else "只报点钟"),
    ) if matches}
    topics = tuple(topic for topic, expression in (("smoke", _SMOKE_QUESTION), ("coordination", _COORDINATION_QUESTION))
                   if any(expression.search(text) for text in texts))
    return VenueRoleTrait(dataset, case_id, cutoff, channel, actor_alias,
        tuple(ref for ref, _, _ in samples), shape, punctuation, next(iter(clocks)) if len(clocks) == 1 else None,
        topics, tuple(interval for _, _, interval in samples if interval is not None))


def blend_venue_role_traits(
    people: Sequence[Persona], *, seed: int, traits: Sequence[VenueRoleTrait],
    target_sender_ids: Collection[str] | None = None,
) -> VenueRoleBlend:
    """给指定合成角色分配弱表达线索，人物身份、偏好和关系由既有人设提供。"""
    actor_ids = {person.sender_id for person in people}
    targets = actor_ids if target_sender_ids is None else set(target_sender_ids)
    if len(actor_ids) != len(people) or not targets <= actor_ids or (targets and not traits):
        raise ValueError("角色与已审核来源必须唯一且覆盖指定混合范围")

    populated = sorted((person for person in people if person.sender_id in targets), key=lambda person: person.sender_id)
    sources = sorted(traits, key=lambda trait: (trait.dataset, trait.case_id, trait.cutoff, trait.actor_alias))
    randomizer = random.Random(f"venue-role-traits:v1:{seed}")
    randomizer.shuffle(populated)
    randomizer.shuffle(sources)
    occupied_styles = Counter(person.speech_profile for person in people)
    updates, audit = {}, {}
    for index, person in enumerate(populated):
        trait = sources[index % len(sources)]
        style = person.speech_profile
        changed = ()
        for dimension, value in (("time_style", trait.time_style), ("punctuation", trait.punctuation_style)):
            if value is None or getattr(style, dimension) == value:
                continue
            candidate = replace(style, **{dimension: value})
            if occupied_styles[candidate] == 0:
                # 只应用一个有证据的维度，避免把原本不同的角色折成同一套表达。
                occupied_styles[style] -= 1
                occupied_styles[candidate] += 1
                style, changed = candidate, (dimension,)
                break

        hints = []
        if trait.message_shape == "brief":
            hints.append("当前条件简单时偏向一条短消息")
        elif trait.message_shape == "multi_clause":
            hints.append("几个相关条件需要一起说时可以用分句")
        if "smoke" in trait.question_topics:
            hints.append("烟况确实影响本次选择时可以先问这一项")
        if "coordination" in trait.question_topics:
            hints.append("人数或时段关系到本次安排时可以只问所缺信息")
        description = style.describe() if changed else person.speaking_style
        updates[person.sender_id] = replace(person, speech_profile=style, speaking_style="。".join((description, *hints)))
        audit[person.sender_id] = VenueRoleAttribution(person.sender_id, trait, changed)

    return VenueRoleBlend(tuple(updates.get(person.sender_id, person) for person in people),
                         tuple(audit[person.sender_id] for person in people if person.sender_id in audit))
