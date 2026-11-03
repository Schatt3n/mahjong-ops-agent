"""看板公开玩法名覆盖当前建局与群报局协议，未知类型不裸露内部值。"""

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BoardEngine, GroupRoomPolicy


@pytest.mark.parametrize("requirement, expected", [
    ({"game_type": "national_standard_mahjong"}, "国标"),
    ({"game_type": "hangzhou_mahjong"}, "杭麻"),
    ({"game_type": "sichuan_mahjong"}, "川麻"),
    ({"game_type": "red_center_mahjong"}, "红中"),
    ({"game_type": "hangzhou_mahjong", "game_variant": "caiqiao"}, "cq"),
    ({"requested_game": "hangzhou_mahjong", "game_variant": "caiqiao"}, "cq"),
    ({"requested_game": "sichuan_mahjong"}, "川麻"),
    ({"game_type": "national_standard_mahjong", "requested_game": "sichuan_mahjong"}, "国标"),
    ({"game_type": "club_custom_mahjong", "requested_game": "national_standard_mahjong"}, "玩法待确认"),
    ({"game_type": "club_custom_mahjong", "game_variant": "caiqiao"}, "玩法待确认"),
    ({"game_type": "PRIVATE_RAW_ENUM"}, "玩法待确认"),
    ({}, "玩法待确认"),
])
def test_public_game_name_uses_canonical_type_and_only_known_names(requirement, expected):
    assert BoardEngine._display_game_name(requirement) == expected


class FakeMessenger:
    def __init__(self):
        self.messages = []

    def send_group_message(self, room_id, text, *, metadata=None):
        self.messages.append(text)
        return f"sent-{len(self.messages)}"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_ordinary_private_national_standard_board_has_public_game_stake_and_seats_without_identity(tmp_path, backend):
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "public-name.sqlite3")
    try:
        store.upsert_group_room_policy(GroupRoomPolicy(room_id="board", outbound_enabled=True))
        game, _ = store.create_game(conversation_id="private-customer", organizer_id="PRIVATE_PARTICIPANT_ID",
            organizer_name="PRIVATE_PARTICIPANT_NAME", requirement={
                "game_type": "national_standard_mahjong", "stake": "1", "cap_score": 256,
                "start_time_kind": "asap_when_full", "smoke_preference": "any",
                "internal_note": "PRIVATE_NOTE",
            }, known_players=[{"customer_id": "PRIVATE_PARTICIPANT_ID", "display_name": "PRIVATE_PARTICIPANT_NAME",
                               "seat_count": 1}], trace_id="trace-create")
        messenger = FakeMessenger()
        snapshot = BoardEngine(store=store, messenger=messenger).publish("board", trace_id="trace-public",
            publication_id="persisted-publication")
        assert snapshot is not None and snapshot.items[0].game_id == game.game_id
        assert len(messenger.messages) == 1
        text = messenger.messages[0]
        assert "国标" in text and "1-256" in text and "173" in text
        assert "PRIVATE_" not in text and "national_standard_mahjong" not in text
    finally:
        if isinstance(store, SQLiteAgentStore):
            store._connection.close()
