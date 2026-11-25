"""验证角色表达线索来自真实可见前缀，并保持独立身份和原有偏好。"""

from dataclasses import replace
import json

import pytest

from . import venue_role_traits
from .venue_actors import ActorContext, generate_personas
from .venue_history_seed import DEFAULT_HISTORY_CASE_ID, DEFAULT_HISTORY_DATASET
from .venue_role_traits import blend_venue_role_traits, load_reviewed_venue_role_traits


def test_reviewed_sources_distinguish_questions_quotes_and_unknowns():
    """询问是否有人和已报烟况分开，只有带可见引用的间隔才提供证据。"""
    traits = load_reviewed_venue_role_traits()
    assert len(traits) == 7 and {trait.version for trait in traits} == {1}
    starts = [trait for trait in traits if trait.case_id == DEFAULT_HISTORY_CASE_ID]
    assert [(trait.message_refs, trait.question_topics, trait.quoted_delays_seconds) for trait in starts] == [
        (("m01",), ("coordination",), ()),
        (("m03",), ("smoke",), (772,)),
        (("m04",), (), (125,)),
    ]
    assert all(trait.cutoff == "m04" and trait.message_shape == "brief" for trait in starts)
    assert all("long_term_preferences" in trait.unknowns and "reading_time" in trait.unknowns for trait in traits)
    assert all(trait.dataset.startswith("eval/") for trait in traits)
    speech_time = next(trait for trait in traits if trait.case_id.endswith("smoke_alternative_20261122_001")
                       and trait.message_refs == ("m01",))
    assert speech_time.time_style == "只报点钟"
    private, = [trait for trait in traits if trait.channel == "private"]
    assert private.cutoff == "2" and private.actor_alias == "customer"
    assert len(private.message_refs) == 1 and private.message_refs[0].endswith(":1")
    assert not private.quoted_delays_seconds


def test_future_group_wording_cannot_change_role_seed(tmp_path, monkeypatch):
    """改写守住原审核前缀之后的数据，发言人特征及来源仍完全相同。"""
    baseline = load_reviewed_venue_role_traits()
    case = json.loads(DEFAULT_HISTORY_DATASET.read_text(encoding="utf-8"))
    case["messages"][4]["text"] = "19:45以后有空位吗？无烟呢？"
    mutated = tmp_path / "reviewed timeline.jsonl"
    mutated.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    load_group = venue_role_traits.load_historical_group_seed

    def with_changed_continuation(path, *, case_id, cutoff):
        return load_group(mutated if case_id == DEFAULT_HISTORY_CASE_ID else path,
                          case_id=case_id, cutoff=cutoff)

    monkeypatch.setattr(venue_role_traits, "load_historical_group_seed", with_changed_continuation)
    assert load_reviewed_venue_role_traits() == baseline


def test_500_synthesized_profiles_remain_distinct_with_reproducible_attribution():
    """覆盖所有来源且保留五百种原来不同的表达组合和独立身份。"""
    people = generate_personas(73)
    traits = load_reviewed_venue_role_traits()
    blend = blend_venue_role_traits(people, seed=91, traits=traits)
    assert blend == blend_venue_role_traits(people, seed=91, traits=tuple(reversed(traits)))
    assert blend != blend_venue_role_traits(people, seed=92, traits=traits)
    assert len({person.speech_profile for person in blend.people}) == len(people) == 500
    assert len(blend.attributions) == 500 and {record.source for record in blend.attributions} == set(traits)
    assert [record.actor_id for record in blend.attributions] == [person.sender_id for person in people]
    assert any(record.speech_dimensions_changed for record in blend.attributions)
    for original, blended in zip(people, blend.people, strict=True):
        assert replace(blended, speech_profile=original.speech_profile, speaking_style=original.speaking_style) == original


def test_only_selected_actors_receive_hints_and_only_audit_receives_provenance():
    """留出组仍保持原人物，模型端看不到原句、源会话或发言人别名。"""
    people = generate_personas(13, count=6)
    target = people[2]
    traits = load_reviewed_venue_role_traits()
    source = next(trait for trait in traits if trait.case_id == DEFAULT_HISTORY_CASE_ID and trait.message_refs == ("m03",))
    blend = blend_venue_role_traits(people, seed=5, traits=(source,), target_sender_ids={target.sender_id})
    assert blend.people[:2] == people[:2] and blend.people[3:] == people[3:]
    assert blend.people[2] != target
    audit, = blend.attributions
    assert audit.actor_id == target.sender_id and audit.to_dict()["source"]["unknowns"] == source.unknowns
    model_profile = ActorContext(blend.people[2], "2026-11-25T19:00:00+08:00").to_dict()["persona"]
    serialized = json.dumps(model_profile, ensure_ascii=False)
    assert "烟况确实影响本次选择时可以先问这一项" in model_profile["speaking_style"]
    assert all(key not in serialized for key in (source.case_id, source.actor_alias, source.dataset, "无烟呢", "772"))
    assert target.smoking == model_profile["smoking"]
    with pytest.raises(ValueError, match="已审核来源"):
        blend_venue_role_traits(people, seed=5, traits=traits, target_sender_ids={"wrong-id"})
