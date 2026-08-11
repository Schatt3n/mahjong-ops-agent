"""验证生产代码持续具备可供团队阅读的中文职责说明。"""

from __future__ import annotations

import runpy
from pathlib import Path


def test_key_boundaries_have_meaningful_chinese_docstrings() -> None:
    """模块、类和桥接函数必须具备有信息量的中文职责说明。"""

    repository_root = Path(__file__).resolve().parents[1]
    checker = runpy.run_path(str(repository_root / "scripts" / "check_chinese_docstrings.py"))
    violations = checker["collect_violations"]()
    assert violations == [], "\n" + "\n".join(violations)
