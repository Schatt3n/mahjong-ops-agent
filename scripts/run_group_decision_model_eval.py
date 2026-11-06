#!/usr/bin/env python3
"""隔离评估本地决策模型；不连接业务运行时、数据库、消息通道或任何写工具。

固定人工标签与本人上下文，单次预测保留首轮错误。Laya 与 LocalJev 使用相同
输入和三个有限选择问题；置信度仅作统计，不能转化成业务授权。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import platform
import statistics
import time
from urllib.parse import urlparse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = ROOT/'eval/golden/group_decision_routing_20261105.jsonl'
MODEL_REVISION = 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
INTENT_CRITERIA = {
    'query': '仅查询可选局或空位，尚未委托招募',
    'delegate': '明确委托寻找协调，尚未选定具体局',
    'create': '明确要求新建或组织一桌局',
    'claim': '明确报名参加某个具体已有局',
    'update': '补充或改变本人既有需求或安排',
    'cancel': '撤销本人需求、等待或参加',
    'chitchat': '闲聊、引述讨论，不推进麻将业务',
    'uncertain': '不能可靠确定本次业务意图',
}
INTENTS = set(INTENT_CRITERIA)


def load_cases(path: Path = DEFAULT_ASSET) -> list[dict]:
    """先检查整套资产和有限选择标签，避免缺失期望被算成通过。"""
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('nonempty unique case ids required')
    for case in cases:
        request = build_request(case)
        for key in ('intent', 'topic', 'own_goal'):
            values = case['expected'].get(f'{key}_any')
            if not isinstance(values, list) or not values or not set(values) <= set(request['questions'][key]['criteria']):
                raise ValueError(f'invalid expected {key}: {case["case_id"]}')
    return cases


def build_request(case: dict) -> dict:
    """只传当前原话、公共卡片和本人必要历史；不读取标签、备注或其他人原文。"""
    data = case['input']
    if set(data) != {'message', 'public_topics', 'own_tasks', 'own_history'}:
        raise ValueError('input fields do not match')
    customer = data['message']['customer_id']
    if set(data['message']) != {'customer_id', 'text'} or not data['message']['text']:
        raise ValueError('message fields do not match')
    if len(data['public_topics']) > 5 or len(data['own_tasks']) > 5 or len(data['own_history']) > 4:
        raise ValueError('candidate or own context budget exceeded')
    for row in data['own_history']:
        if set(row) != {'customer_id', 'text'} or row['customer_id'] != customer:
            raise ValueError('own_history includes foreign or unprojected fields')
    for row in data['public_topics']:
        if set(row) != {'topic_id', 'game_type', 'stakes', 'summary'}:
            raise ValueError('public_topics includes unprojected fields')
    for row in data['own_tasks']:
        if set(row) != {'goal_id', 'topic_id', 'summary', 'status'}:
            raise ValueError('own_tasks includes unprojected fields')
    topic = {row['topic_id']: f"{row['game_type']}{'/'.join(row['stakes'])}的公共话题" for row in data['public_topics']}
    topic.update({'new':'新建公共话题，候选没有相同业务', 'none':'本条不关联业务话题', 'uncertain':'现有信息无法唯一确定话题'})
    goals = {row['goal_id']: '本人此前已存在的这个业务目标' for row in data['own_tasks']}
    goals.update({'new':'本人的新目标，不接续已有任务', 'none':'本条没有个人业务目标', 'uncertain':'无法唯一确定本人的目标'})
    questions = {
        'intent': {'type':'choice', 'instructions':'判断当前message的实际意图。结合本人历史，保留否定和引述范围。条件齐全不是建局授权。更新已有目标与新查询不同。', 'criteria':INTENT_CRITERIA},
        'topic': {'type':'choice', 'instructions':'当前消息关联哪个公共话题？不同玩法即使同档位也要分开。不能拿别人的公共话题当成自己的既有安排。无法唯一指代时选uncertain。', 'criteria':topic},
        'own_goal': {'type':'choice', 'instructions':'当前消息延续本人哪个既有目标？同一话题可以有两个不同目标，不默认最新。明确另一个目标选new；没有充分指代选uncertain；闲聊选none。', 'criteria':goals},
    }
    return {'state':deepcopy(data), 'questions':deepcopy(questions)}


def score_response(case: dict, response: dict, *, latency_ms: float) -> dict:
    """独立统计分类、误关联及高置信错误；无效或截断输出不能默认为正确。"""
    request = build_request(case)
    answers, usage = response.get('answers', {}), response.get('usage', {})
    errors = []
    if usage.get('truncated') or usage.get('state_tokens_dropped') or usage.get('truncated_questions') or usage.get('options'):
        errors.append('input_truncated')
    result = {'case_id':case['case_id'], 'latency_ms':latency_ms, 'side_effects_authorized':False,
              'predicted':{}, 'confidence':{}, 'correctness':{}}
    for key in ('intent', 'topic', 'own_goal'):
        answer = answers.get(key, {})
        chosen = answer.get('choice')
        if chosen not in request['questions'][key]['criteria']:
            errors.append(f'invalid_{key}_choice')
        value = answer.get('answer_confidence', answer.get('confidence'))
        confidence = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1 else None
        result['predicted'][key] = chosen
        result['confidence'][key] = confidence
        result['correctness'][key] = chosen in case['expected'][f'{key}_any']
        result[f'{key}_correct'] = result['correctness'][key]
    result['valid'] = not errors
    result['errors'] = errors
    result['passed'] = result['valid'] and all(result['correctness'].values())
    result['misassociation'] = any(not result['correctness'][key] and result['predicted'][key] not in {None, 'new', 'none', 'uncertain'} for key in ('topic', 'own_goal'))
    result['high_confidence_error'] = any(not result['correctness'][key] and (result['confidence'][key] or 0) >= 0.85 for key in ('intent', 'topic', 'own_goal'))
    return result


def validate_local_endpoint(endpoint: str) -> str:
    """只允许显式本机端点，避免本地实验暗中外发给远端模型或携带凭证。"""
    parsed = urlparse(endpoint)
    try:
        loopback = parsed.hostname == 'localhost' or ipaddress.ip_address(parsed.hostname).is_loopback
    except (ValueError, TypeError):
        loopback = False
    if not loopback or parsed.scheme != 'http' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('only credential-free loopback HTTP endpoints are allowed')
    return endpoint.rstrip('/')


def localjev_payload(request: dict) -> dict:
    """LocalJev仅改变传输封套，保持与本地Laya相同的文本及有限选择问题。"""
    return {'model':'jev-latest', **deepcopy(request)}


def check_laya_encoding(agent, request: dict) -> dict:
    """按锁定版本真实tokenizer预检正文、指令及每个选项，拒绝库内隐式截断。

    Laya 0.4.22即使max_len足够仍会把每个选项限制为48 tokens，因此逐项核对；
    这属于输入完整性检查，不能通过删掉当前用户原话来满足限制。
    """
    from laya.common import render_options, build_head, encode_text
    ids = list(request['questions'])
    internal = {key:agent._to_internal(request['questions'][key]) for key in ids}
    for key, question in internal.items():
        for option in render_options(question):
            if len(encode_text(agent.tok, ' '+option, add_special_tokens=False)['input_ids']) > 48:
                raise ValueError(f'option exceeds library 48-token cap: {key}')
        head, _, stats = build_head(agent.tok, question, 2048)
        instruction = encode_text(agent.tok, f"{question['t']} question: {question['ins']}", add_special_tokens=False)['input_ids']
        if stats['tokens_per_option'] is not None or len(instruction) > 2048 - sum(1+len(encode_text(agent.tok, ' '+option, add_special_tokens=False)['input_ids']) for option in render_options(question)):
            raise ValueError(f'question head truncated: {key}')
    encoded = agent._encode_state(request['state'], ids, internal, max_len=8192, head_max_len=2048)
    if any(item['state_stats']['state_tokens_dropped'] for item in encoded):
        raise ValueError('state truncated')
    return {'state_tokens':encoded[0]['state_stats']['state_tokens'],
            'max_encoded_tokens':max(len(item['ids']) for item in encoded), 'input_truncated':False}


def summarize(results: list[dict]) -> dict:
    """分别报告首轮冷预测与其余热延迟；错误率仅针对这套小样本，不宣称总体准确率。"""
    valid = [row for row in results if row.get('valid')]
    warm = sorted(row['latency_ms'] for row in valid[1:])
    return {'case_count':len(results), 'valid_count':len(valid), 'passed_count':sum(row['passed'] for row in results),
        'correct_counts':{key:sum(row.get(f'{key}_correct', False) for row in results) for key in ('intent','topic','own_goal')},
        'misassociation_count':sum(row.get('misassociation', False) for row in results),
        'high_confidence_error_count':sum(row.get('high_confidence_error', False) for row in results),
        'first_prediction_ms':valid[0]['latency_ms'] if valid else None,
        'warm_p50_ms':statistics.median(warm) if warm else None,
        'warm_p95_ms':warm[min(len(warm)-1, int(len(warm)*0.95))] if warm else None,
        'failed_case_ids':[row['case_id'] for row in results if not row['passed']]}


def main() -> int:
    """在显式独立输出目录运行一次；不覆盖既往报告，也不自动重试失败预测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['laya','localjev'], required=True)
    parser.add_argument('--asset', type=Path, default=DEFAULT_ASSET)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=ROOT/'runtime_data/decision_model_trial_20261105')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    cases = load_cases(args.asset)
    requests = [build_request(case) for case in cases]
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'backend':args.backend, 'dataset_sha256':hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        'request_sha256':hashlib.sha256(json.dumps(requests,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        'context_mode':'fixed human-annotated prior state, not cascading model predictions',
        'high_confidence_threshold':0.85, 'production_integration':False, 'automatic_retries':0,
        'platform':platform.machine(), 'case_count':len(cases)}
    (args.output/'requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2))
    if args.backend == 'laya':
        # 仅进程内指定实验缓存，不写项目.env、主虚拟环境或全局配置。
        for name, folder in [('HF_HOME','hf_cache'),('TORCH_HOME','torch_cache'),('XDG_CACHE_HOME','cache')]:
            os.environ[name] = str(args.cache_root/folder)
        started = time.perf_counter()
        import torch
        from laya import Router
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; do not silently change test device')
        router = Router(device='mps',standalone_repos=True,revisions={'multilingual':MODEL_REVISION},max_loaded=1)
        agent = router.load('multilingual')
        metadata.update({'versions':{name:importlib.metadata.version(name) for name in ('laya','torch','transformers')},
            'model':'convaiinnovations/laya-multilingual','revision':MODEL_REVISION,'device':str(agent.device),
            'cached_load_seconds':time.perf_counter()-started,'max_len':8192,'head_max_len':2048,
            'confidence_semantics':'answer_confidence from local logits; threshold not calibrated on this dataset'})
        encodings = [check_laya_encoding(agent, request) for request in requests]
        metadata['input_encoding'] = {'max_state_tokens':max(r['state_tokens'] for r in encodings),
            'max_encoded_tokens':max(r['max_encoded_tokens'] for r in encodings),'all_untruncated':True}
        def predict(request):
            """只向已加载本地模型提交相同封套，设备同步保证计时完整。"""
            response = router.predict(**request,model='multilingual',lang='zh',max_len=8192,head_max_len=2048)
            torch.mps.synchronize()
            return response
    else:
        endpoint = validate_local_endpoint(args.endpoint)
        metadata['confidence_semantics'] = 'generated/self-reported probabilities; not direct logit read'
        try:
            with urllib.request.urlopen(endpoint+'/ready', timeout=3) as response:
                if response.status != 200:
                    raise RuntimeError('LocalJev readiness failed')
        except Exception as exc:
            metadata.update({'status':'not_run','reason':'No ready local LocalJev endpoint','error_type':type(exc).__name__})
            (args.output/'report.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
            print(json.dumps({'status':'not_run','report':str(args.output/'report.json')}))
            return 2
        def predict(request):
            """仅调用本机Jev适配层，不自行切换远端或换用另一模型。"""
            data = json.dumps(localjev_payload(request),ensure_ascii=False).encode()
            req = urllib.request.Request(endpoint+'/v1/systemone',data=data,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as response:
                return json.loads(response.read())
    (args.output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    results = []
    for case, request in zip(cases, requests):
        started = time.perf_counter()
        try:
            raw = predict(request)
            result = score_response(case, raw, latency_ms=(time.perf_counter()-started)*1000)
            result['raw_response'] = raw
        except Exception as exc:
            result = {'case_id':case['case_id'],'passed':False,'valid':False,'latency_ms':(time.perf_counter()-started)*1000,
                      'error_type':type(exc).__name__,'error':str(exc)[:500],'side_effects_authorized':False}
        results.append(result)
        with (args.output/'results.jsonl').open('a') as file:
            file.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps({key:result.get(key) for key in ('case_id','passed','latency_ms','predicted','error_type')},ensure_ascii=False),flush=True)
    report = {**metadata, 'status':'passed' if all(row['passed'] for row in results) else 'failed', **summarize(results)}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
