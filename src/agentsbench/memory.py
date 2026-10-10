"""Persistent, judge-scoped memories of completed case deliberations."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .core import Case, Judge, Opinion


def _normalize(value: str) -> str:
    return "".join(value.split()).casefold()


class MemoryStore:
    def __init__(self, path: Path, recall_limit: int = 3, max_per_judge: int = 100):
        if recall_limit < 1 or max_per_judge < 1:
            raise ValueError("memory limits must be positive")
        self.path = Path(path)
        self.recall_limit = recall_limit
        self.max_per_judge = max_per_judge
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("records"), list):
                raise ValueError(f"unsupported memory format: {self.path}")
            self.records = data["records"]
        else:
            self.records = []

    def recall(self, judge: Judge, case: Case) -> list[dict]:
        statutes = _normalize(case.statutes)
        candidates = []
        for index, record in enumerate(self.records):
            if record["judge"] != judge.name or record["case_id"] == case.case_id:
                continue
            same_charge = bool(case.charge and record["charge"] and _normalize(case.charge) == _normalize(record["charge"]))
            same_statutes = bool(statutes and statutes == _normalize(record["statutes"]))
            if same_charge or same_statutes:
                candidates.append((10 * same_charge + 5 * same_statutes, index, record))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            {
                "case_id": record["case_id"],
                "kind": record["kind"],
                "months": record["months"],
                "rationale": record["rationale"],
            }
            for _, _, record in candidates[: self.recall_limit]
        ]

    def remember(self, case: Case, judges: list[Judge], presiding: Judge, opinions: dict[str, Opinion], final: Opinion) -> None:
        selected = {judge.name for judge in judges}
        # Re-running a case replaces its old memories; it never trains on itself.
        self.records = [
            record for record in self.records
            if not (record["case_id"] == case.case_id and record["judge"] in selected)
        ]
        for judge in judges:
            is_presiding = judge.name == presiding.name
            opinion = final if is_presiding else opinions[judge.name]
            self.records.append({
                "judge": judge.name,
                "case_id": case.case_id,
                "charge": case.charge,
                "statutes": case.statutes,
                "kind": "legal_reasoning" if is_presiding else "social_observation",
                "months": opinion.months,
                "rationale": opinion.rationale[:1000],
            })
        for name in selected:
            indices = [i for i, record in enumerate(self.records) if record["judge"] == name]
            excess = len(indices) - self.max_per_judge
            if excess > 0:
                remove = set(indices[:excess])
                self.records = [record for i, record in enumerate(self.records) if i not in remove]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps({"version": 1, "records": self.records}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
