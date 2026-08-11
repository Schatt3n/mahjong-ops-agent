#!/usr/bin/env python3
"""检查生产源码中的关键边界是否具备有信息量的中文说明。"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPOSITORY_ROOT / "src" / "mahjong_agent_runtime",
    REPOSITORY_ROOT / "scripts",
    REPOSITORY_ROOT / "integrations",
)
CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]")
JAVASCRIPT_FUNCTION_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(",
    re.MULTILINE,
)
LOW_VALUE_DOCSTRING_PATTERNS = (
    re.compile(r"实现麻将运营智能体的.+能力"),
    re.compile(r"提供.+命令行运维或验证入口"),
    re.compile(r"集中管理.+的生命周期、协作边界和异常处理"),
    re.compile(r"执行.+相关处理，并遵守当前模块定义的状态与安全边界"),
    re.compile(r"内部辅助方法：完成"),
    re.compile(r"相关的数据与行为"),
    re.compile(r"结构化数据合同"),
    re.compile(r"初始化对象依赖；构造阶段不执行不可逆的外部业务动作"),
)


def has_chinese_docstring(node: ast.AST) -> bool:
    """判断 AST 节点的正式文档字符串是否包含中文。"""

    value = ast.get_docstring(node, clean=False)
    return bool(value and CHINESE_PATTERN.search(value))


def low_value_docstring(node: ast.AST) -> str | None:
    """返回命中的低信息量模板；未命中时返回 ``None``。"""

    value = ast.get_docstring(node, clean=True) or ""
    for pattern in LOW_VALUE_DOCSTRING_PATTERNS:
        if pattern.search(value):
            return pattern.pattern
    return None


def javascript_preceding_comment(lines: list[str], function_line_index: int) -> str:
    """读取 JavaScript 函数前紧邻的 JSDoc 或连续行注释。"""

    index = function_line_index - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0:
        return ""

    stripped = lines[index].strip()
    if stripped.startswith("//"):
        comments: list[str] = []
        while index >= 0 and lines[index].strip().startswith("//"):
            comments.append(lines[index].strip())
            index -= 1
        return "\n".join(reversed(comments))

    if not stripped.endswith("*/"):
        return ""
    comments = []
    while index >= 0:
        comments.append(lines[index])
        if "/**" in lines[index] or "/*" in lines[index]:
            break
        index -= 1
    return "\n".join(reversed(comments))


def collect_javascript_violations(path: Path) -> list[str]:
    """检查桥接层 JavaScript 函数是否有中文职责说明。"""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    relative = path.relative_to(REPOSITORY_ROOT)
    violations: list[str] = []
    for match in JAVASCRIPT_FUNCTION_PATTERN.finditer(text):
        line_number = text.count("\n", 0, match.start()) + 1
        comment = javascript_preceding_comment(lines, line_number - 1)
        if not CHINESE_PATTERN.search(comment):
            violations.append(f"{relative}:{line_number} JavaScript 函数 {match.group('name')} 缺少中文说明")
    return violations


def collect_violations() -> list[str]:
    """收集缺失中文边界说明或仍使用低价值模板的位置。"""

    violations: list[str] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(REPOSITORY_ROOT)
            if not has_chinese_docstring(tree):
                violations.append(f"{relative}:1 模块缺少中文说明")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = getattr(node, "name", "<module>")
                line_number = getattr(node, "lineno", 1)
                docstring = ast.get_docstring(node, clean=False)

                # 模块和类是稳定的工程边界，必须解释职责。函数是否需要注释由
                # 复杂度决定；已有文档字符串则必须是中文且不能是占位模板。
                if isinstance(node, ast.ClassDef) and not has_chinese_docstring(node):
                    violations.append(f"{relative}:{line_number} 类 {name} 缺少中文职责说明")
                    continue
                if docstring and not has_chinese_docstring(node):
                    violations.append(f"{relative}:{line_number} {name} 的文档字符串缺少中文说明")
                    continue
                pattern = low_value_docstring(node)
                if pattern:
                    violations.append(
                        f"{relative}:{line_number} {name} 使用低信息量模板：{pattern}"
                    )
        for path in sorted(root.rglob("*.mjs")):
            if "node_modules" in path.parts:
                continue
            violations.extend(collect_javascript_violations(path))
    return violations


def main() -> int:
    """运行检查；发现缺失或模板化说明时打印位置并返回非零状态码。"""

    violations = collect_violations()
    if not violations:
        print("中文文档检查通过")
        return 0
    print("\n".join(violations))
    print(f"共发现 {len(violations)} 处中文文档质量问题")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
