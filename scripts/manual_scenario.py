"""手动场景的纯输入合同；全量校验完成前不创建数据库、不请求模型。"""

from __future__ import annotations

from copy import deepcopy
import re

from mahjong_agent_runtime import CustomerProfile
from mahjong_agent_runtime.knowledge import default_terminology_repository


PROFILE_FIELDS = {
    'display_name', 'gender', 'preferred_games', 'preferred_stakes', 'preferred_time_tags',
    'profile_facts', 'smoke_preference', 'private_remark', 'notes', 'no_contact',
}


def validated_profile(changes: dict, *, actor_id: str, before: dict | None = None) -> CustomerProfile:
    """共用画像白名单；仅规范化门店玩法别名，原始场景由导入清单独立保存。"""
    if not isinstance(changes, dict) or set(changes) - PROFILE_FIELDS:
        raise ValueError('画像字段无效')
    values = deepcopy(before) if before is not None else CustomerProfile(customer_id=actor_id, display_name='新客户').to_dict()
    for key, value in changes.items():
        if key in {'preferred_games', 'preferred_stakes', 'preferred_time_tags', 'profile_facts'}:
            if not isinstance(value, list) or len(value) > 12 or any(not isinstance(item, str) or not item.strip() or len(item) > 240 for item in value):
                raise ValueError('画像列表最多12项，每项须为1至240字')
            values[key] = list(dict.fromkeys(item.strip() for item in value))
        elif key == 'no_contact':
            if not isinstance(value, bool):
                raise ValueError('触达开关须为布尔值')
            values[key] = value
        elif key == 'smoke_preference':
            if value is not None and (not isinstance(value, str) or value not in {'any', 'no_smoke', 'smoking'}):
                raise ValueError('烟况须为未知、不限、无烟或有烟')
            values[key] = value
        elif key == 'gender':
            if value is not None and (not isinstance(value, str) or value not in {'未知', '男', '女'}):
                raise ValueError('性别值无效')
            values[key] = value
        else:
            if not isinstance(value, str) or len(value) > 1000:
                raise ValueError('画像文本须在1000字以内')
            values[key] = value.strip()
    name = values['display_name']
    if not name or len(name) > 30:
        raise ValueError('角色名须为1至30字')
    normalized = []
    for game in values['preferred_games']:
        term = next((term for term in default_terminology_repository().terms if term.category == 'game_type'
                     and game.casefold() in {alias.casefold() for alias in term.aliases}), None)
        normalized.append(str(term.canonical['requested_game']) if term else game)
    values['preferred_games'] = list(dict.fromkeys(normalized))
    values['public_name'] = name
    return CustomerProfile(**values)


def validate_scenario(scenario: dict) -> dict:
    """拒绝整包中的未知字段、错误引用与超限输入；不静默截断消息或修正角色身份。"""
    if not isinstance(scenario, dict) or set(scenario) != {'version', 'actors', 'messages'}:
        raise ValueError('场景仅接受version、actors、messages三个字段')
    if type(scenario['version']) is not int or scenario['version'] != 1:
        raise ValueError('场景version须为整数1')
    actors, messages = scenario['actors'], scenario['messages']
    if not isinstance(actors, list) or not 1 <= len(actors) <= 20:
        raise ValueError('场景须包含1至20位角色')
    profiles = []
    ids, names = set(), set()
    for actor in actors:
        if not isinstance(actor, dict) or set(actor) != {'id', 'profile', 'is_friend'}:
            raise ValueError('角色须且仅可包含id、profile、is_friend')
        actor_id = actor['id']
        if not isinstance(actor_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', actor_id):
            raise ValueError('角色id须为1至64位字母、数字、下划线或连字符')
        if actor_id in ids:
            raise ValueError('角色id不能重复')
        if not isinstance(actor['is_friend'], bool):
            raise ValueError('is_friend须为布尔值')
        if not isinstance(actor['profile'], dict) or 'display_name' not in actor['profile']:
            raise ValueError('每位角色须提供profile.display_name')
        profile = validated_profile(actor['profile'], actor_id=actor_id)
        if profile.display_name in names:
            raise ValueError('角色名不能重复')
        ids.add(actor_id)
        names.add(profile.display_name)
        profiles.append({'id': actor_id, 'profile': profile, 'is_friend': actor['is_friend']})
    if not isinstance(messages, list) or not 1 <= len(messages) <= 30:
        raise ValueError('场景须包含1至30条消息')
    for message in messages:
        if not isinstance(message, dict) or set(message) != {'actor_id', 'channel', 'text'}:
            raise ValueError('消息须且仅可包含actor_id、channel、text')
        if not isinstance(message['actor_id'], str) or message['actor_id'] not in ids:
            raise ValueError('消息引用了不存在的角色')
        if not isinstance(message['channel'], str) or message['channel'] not in {'group', 'private'}:
            raise ValueError('消息通道须为group或private')
        if not isinstance(message['text'], str) or not message['text'].strip() or len(message['text']) > 4000:
            raise ValueError('消息正文须为1至4000字且不能全为空白')
    return {'actors': profiles, 'messages': deepcopy(messages)}
