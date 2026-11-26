"""从真实送达账本和逐条核对记录生成错报统计，未判定样本保留在分母覆盖情况中。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


_VERDICTS = {"no_issue_found", "factual_mismatch", "unsupported_claim", "uncertain"}
_SOURCES = {"model", "deterministic", "unknown"}


def summarize_cohort(cohort: dict) -> dict:
    """按送达顺序冻结范围，避免仅选有问题的消息、重复计数或把未核对算通过。"""
    if cohort["role"] not in {"agent", "customer"}:
        raise ValueError("role must be agent or customer")
    first = cohort.get("from_sequence", 1)
    cutoff = cohort["through_sequence"]
    if type(first) is not int or type(cutoff) is not int or not 1 <= first <= cutoff:
        raise ValueError("delivery sequence bounds must be positive integers in ascending order")
    archive = Path(cohort["run_directory"]).resolve() / "chat.sqlite3"
    timer_due = cohort.get("timer_due_at")
    timer_ids = None
    if timer_due:
        if cohort["role"] != "customer":
            raise ValueError("a timer wave is a customer cohort")
        due = datetime.fromisoformat(timer_due)
        if due.utcoffset() is None:
            raise ValueError("timer_due_at must include a timezone")
        journal = archive.parent / "role_events.sqlite3"
        with closing(sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True)) as connection:
            timer_events = connection.execute(
                "SELECT event_id, status FROM venue_role_events WHERE kind = 'timer' AND due_at = ?",
                (due.astimezone(timezone.utc).isoformat(timespec="microseconds"),),
            ).fetchall()
        if not timer_events or any(status != "finished" for _, status in timer_events):
            raise ValueError("timer wave is missing or still in progress")
        timer_ids = {f"customer:{event_id}" for event_id, _ in timer_events}
    # 定时事件在消息持久化后才完成。先读完结批次、再读送达，避免两套库的读取顺序漏计刚结束的投递。
    with closing(sqlite3.connect(archive.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT sequence, message_id, trace_id, sent_at FROM venue_messages WHERE sequence BETWEEN ? AND ? "
            + ("AND direction = 'outbound' AND sender_id = 'simulation-agent' " if cohort["role"] == "agent"
               else "AND direction IN ('inbound', 'peer') AND sender_id <> 'simulation-agent' ")
            + "ORDER BY sequence", (first, cutoff),
        ).fetchall()
    if timer_ids is not None:
        rows = [row for row in rows if row["message_id"] in timer_ids]

    assessments = {entry["sequence"]: entry for entry in cohort["assessments"]}
    if len(assessments) != len(cohort["assessments"]) or not assessments.keys() <= {row["sequence"] for row in rows}:
        raise ValueError("assessments must identify unique deliveries in the frozen cohort")
    audit = []
    for row in rows:
        datum = assessments.get(row["sequence"])
        if datum is None:
            audit.append({**dict(row), "source": "unknown", "verdict": "unreviewed"})
            continue
        if (datum["source"] not in _SOURCES or datum["verdict"] not in _VERDICTS
                or not isinstance(datum.get("evidence"), list) or not datum["evidence"]):
            raise ValueError("each assessment must state its source, verdict and evidence")
        audit.append({**dict(row), **{key: datum[key] for key in ("source", "verdict", "evidence")},
                      **({"note": datum["note"]} if "note" in datum else {})})

    return {"label": cohort["label"], "role": cohort["role"], "run_directory": str(archive.parent),
            "from_sequence": first, "through_sequence": cutoff, "all_delivered": summarize_messages(audit),
            **({"timer_due_at": timer_due} if timer_due else {}),
            "sources": {source: sum(row["source"] == source for row in audit) for source in sorted(_SOURCES)},
            "model_delivered": summarize_messages([row for row in audit if row["source"] == "model"]),
            "deliveries": audit}


def summarize_messages(messages: list[dict]) -> dict:
    """统计每条消息的最高确定问题，疑似或未审单列，两个比率都给出原始计数。"""
    counts = {key: sum(row["verdict"] == key for row in messages) for key in (*sorted(_VERDICTS), "unreviewed")}
    total = len(messages)
    resolved = total - counts["uncertain"] - counts["unreviewed"]
    wrong = counts["factual_mismatch"]
    unsupported = counts["unsupported_claim"]
    return {
        "total": total, "resolved": resolved, **counts,
        "factual_mismatch_per_delivery": wrong / total if total else None,
        "confirmed_issue_per_delivery": (wrong + unsupported) / total if total else None,
        "confirmed_issue_among_resolved": (wrong + unsupported) / resolved if resolved else None,
    }


def main() -> int:
    """读取逐条核对清单并输出带验证覆盖率的统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {"cohorts": [summarize_cohort(cohort) for cohort in data["cohorts"]],
              "method": "按冻结范围的真实送达计算样本错报率。模型比率只用来源已核实的模型消息；未核对、来源不明或事实无法判定的记录单列。"}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
