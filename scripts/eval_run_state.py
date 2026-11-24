"""绑定完整评测候选和已归档证据，防止续跑混入变更和并发结果。"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from eval_retry import write_json


def _hash_file(path: Path) -> str:
    """流式摘要原始文件，避免大型轨迹和运行库占满内存。"""
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def candidate_fingerprint(root: Path, run_id: str) -> str:
    """摘要实际代码和有效模型环境，只保存合成摘要，不写配置密钥。"""
    if (root / '.git').exists():
        raw = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
        files = {Path(os.fsdecode(path)) for path in raw.split(b'\0') if path}
    else:
        # 非 Git 发行副本仍以实际可执行文件与评测资产绑定候选。
        files = {p.relative_to(root) for base in ('src', 'scripts', 'tests', 'eval')
                 for p in (root / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        files.update(Path(name) for name in ('AGENTS.md', 'pyproject.toml') if (root / name).is_file())
    rows = [(p.as_posix(), _hash_file(root / p) if (root / p).is_file() else None) for p in sorted(files)]
    env = {name: value for name, value in os.environ.items() if
           name.startswith(('MAHJONG_', 'DEEPSEEK_', 'OPENAI_', 'ARK_')) and not name.startswith('MAHJONG_EVAL_RETRY_')}
    env.update({name: os.environ.get(name) for name in ('PYTHONPATH', 'TZ')})
    return hashlib.sha256(json.dumps([run_id, sys.executable, sys.version, rows, env],
                                     sort_keys=True).encode()).hexdigest()


def evidence_fingerprint(run_dir: Path, scopes: list[dict]) -> str:
    """冻结同轮日志、数据库、尝试账本及引用的外部报告原件。"""
    paths = {p for p in run_dir.rglob('*') if p.is_file() and p.name != 'gate_report.json'
             and not p.name.startswith('.run-lock.sqlite3')}
    for phase in scopes:
        if phase.get('source_report_path'):
            source = Path(phase['source_report_path'])
            if source.is_file() and not source.is_relative_to(run_dir):
                paths.update(p for p in source.parent.rglob('*') if p.is_file())
    rows = [(str(p), _hash_file(p)) for p in sorted(paths)]
    return hashlib.sha256(json.dumps([rows, scopes], sort_keys=True).encode()).hexdigest()


def resume_directory(parent: Path, run_id: str) -> Path:
    """续跑只允许选择当前仓库自己创建的门禁目录。"""
    if not re.fullmatch(r'\d{8}T\d{12}Z_[a-zA-Z0-9_]+', run_id):
        raise ValueError('invalid evaluation run ID')
    run_dir = parent / run_id
    if run_dir.is_symlink() or not (run_dir / 'gate_report.json').is_file():
        raise ValueError('evaluation run not found')
    return run_dir


@contextmanager
def run_lock(run_dir: Path):
    """用跨平台 SQLite 独占锁避免两个续跑命令消费同一账本。"""
    connection = sqlite3.connect(run_dir / '.run-lock.sqlite3', timeout=0)
    try:
        try:
            connection.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as exc:
            raise ValueError('evaluation run already has an active coordinator') from exc
        yield
    finally:
        connection.close()


def resume_report(run_dir: Path, plan: list[dict], fingerprint: str) -> dict:
    """只重新打开已明确受阻且证据完整的同一轮，继续原计划而不拼接历史。"""
    report = json.loads((run_dir / 'gate_report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'blocked' or report.get('full') is not True or report.get('failed_scopes')
            or report.get('run_id') != run_dir.name):
        raise ValueError('only complete-mode runs blocked by verified transport faults can resume')
    if report.get('candidate_fingerprint') != fingerprint:
        raise ValueError('candidate code, evaluation assets or model configuration changed')
    planned = {phase['scope']: phase['command'] for phase in plan}
    if {phase['scope']: phase['command'] for phase in report['scopes']} != planned:
        raise ValueError('evaluation plan changed')
    if report.get('evidence_fingerprint') != evidence_fingerprint(run_dir, report['scopes']):
        raise ValueError('saved evaluation evidence changed')
    if len([phase for phase in report['scopes'] if phase['status'] == 'blocked']) != 1:
        raise ValueError('invalid blocked scope checkpoint')
    # 封存原状态和预算，后续的同轮报告从新窗口继续更新。
    write_json(run_dir / f"gate_report.window_{report['retry_window']:03d}.json", report)
    report['retry_window'] += 1
    report['status'] = 'running'
    report.pop('finished_at_utc', None)
    report.pop('evidence_fingerprint', None)
    for phase in report['scopes']:
        if phase['status'] in {'blocked', 'not_run'}:
            phase['status'] = 'pending'
    return report
