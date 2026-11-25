"""真实扩容须结合实际参与和评测覆盖，防止预装人数冒充真实验收。"""

from scripts.summarize_venue_run import actor_observations, audit_rolling_expansion


def test_expansion_gate_keeps_mock_partial_actors_and_duplicate_reviews_out():
    """同名册的离线完成、角色未覆盖和审阅重复均不能放行下一档。"""
    report = {
        "mode": "real", "scenario": "rolling-roles", "status": "passed", "people": 500, "targeted_users": 12,
        "role_seed_profiles": {"derived_roles": 500, "distinct_speech_profiles": 500},
        "rolling_roles": {"real_decision_actors": 12, "actors_with_two_calendar_days": 12,
            "real_decision_actors_with_two_calendar_days": 12,
            "sent_channels": {"group": 1, "private": 1}, "actors_with_prior_day_private_receipts_in_context": 1,
            "cross_channel_replies": {"group_to_private": 1, "private_to_group": 0},
            "wake_kinds": {"group_attention": 3, "private_receipt": 1, "rain": 1, "traffic": 1}},
        "checks": {"role_context_matches_visible_channel_history": True, "private_receipts_wake_only_their_recipient": True,
                   "no_overfilled_table_in_store_or_tool_evidence": True},
        "semantic_samples": [{"case_id": str(i), "status": "valid", "verdict": "pass"} for i in range(12)],
    }
    assert audit_rolling_expansion(report, 12)["status"] == "passed"
    assert not audit_rolling_expansion(report | {"mode": "mock"}, 12)["checks"]["real_history_grounded_rolling_run"]
    partial = report | {"rolling_roles": report["rolling_roles"] | {"real_decision_actors": 11}}
    assert not audit_rolling_expansion(partial, 12)["checks"]["all_expected_actors_decided_over_two_calendar_days"]
    failed_day = report | {"rolling_roles": report["rolling_roles"] | {"real_decision_actors_with_two_calendar_days": 11}}
    assert not audit_rolling_expansion(failed_day, 12)["checks"]["all_expected_actors_decided_over_two_calendar_days"]
    no_link = report | {"rolling_roles": report["rolling_roles"] | {"cross_channel_replies": {}}}
    assert not audit_rolling_expansion(no_link, 12)["checks"]["actual_cross_channel_response"]
    duplicate = report | {"semantic_samples": report["semantic_samples"][:11] + [report["semantic_samples"][0]]}
    assert not audit_rolling_expansion(duplicate, 12)["checks"]["semantic_review_reached_target"]


def test_archived_timers_and_outages_do_not_count_as_actual_model_days():
    """两天都拿到角色模型的完整响应才算跨日，等待也可以是有效决定。"""
    valid_wait = {"source": "model", "status": "valid", "response_source": {"http_response_complete": True},
                  "action": {"action": "wait"}}
    failed = {"source": "model", "status": "transport_error"}
    first, second = "2026-11-25T19:00:00+08:00", "2026-11-26T19:00:00+08:00"
    events = [{"actor_id": "a", "virtual_at": first, "decision": valid_wait},
              {"actor_id": "a", "virtual_at": second, "decision": valid_wait},
              {"actor_id": "b", "virtual_at": first, "decision": valid_wait},
              {"actor_id": "b", "virtual_at": second, "decision": failed}]
    assert actor_observations(events)["real_decision_actors_with_two_calendar_days"] == 1
