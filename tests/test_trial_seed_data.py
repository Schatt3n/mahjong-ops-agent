from mahjong_agent import SEED_CUSTOMERS


def test_trial_seed_customers_contains_expected_mock_profiles() -> None:
    assert len(SEED_CUSTOMERS) == 50


def test_trial_seed_customer_ids_are_unique() -> None:
    ids = [str(customer["id"]) for customer in SEED_CUSTOMERS]

    assert len(ids) == len(set(ids))


def test_trial_seed_customers_have_store_required_fields() -> None:
    required_fields = {
        "id",
        "display_name",
        "contact",
        "preferred_games",
        "preferred_levels",
        "usual_start_hours",
        "smoke_preference",
        "response_speed",
        "response_rate",
        "notes",
    }

    for customer in SEED_CUSTOMERS:
        assert required_fields <= set(customer)
        assert customer["id"]
        assert customer["display_name"]
        assert isinstance(customer["preferred_games"], list)
        assert isinstance(customer["preferred_levels"], list)
        assert isinstance(customer["usual_start_hours"], list)


def test_trial_seed_keeps_zhang_profile_business_facts() -> None:
    zhang = next(customer for customer in SEED_CUSTOMERS if customer["id"] == "zhang")

    assert zhang["display_name"] == "张哥"
    assert {"杭麻", "川麻"} <= set(zhang["preferred_games"])
    assert {"0.5", "1", "1-32", "2", "2-64"} <= set(zhang["preferred_levels"])
    assert zhang["smoke_preference"] == "any"
    assert zhang["usual_party_size"] == 1
    assert zhang["usual_party_size_confidence"] >= 0.65
    assert "男性" in str(zhang["notes"])
    assert "五番封顶" in str(zhang["notes"])
