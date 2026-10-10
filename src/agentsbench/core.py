"""A small, inspectable implementation of the four-stage paper workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
import re
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import MemoryStore


@dataclass(frozen=True)
class Case:
    case_id: str
    facts: str
    statutes: str
    charge: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "Case":
        for key in ("case_id", "facts", "statutes"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"case requires a nonempty string: {key}")
        if not isinstance(value.get("charge", ""), str):
            raise ValueError("charge must be a string")
        return cls(**{key: value.get(key, "") for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Judge:
    name: str
    role: str
    perspective: str


@dataclass(frozen=True)
class Opinion:
    months: int
    rationale: str

    @classmethod
    def from_text(cls, text: str) -> "Opinion":
        # Some OpenAI-compatible models prepend a tagged reasoning segment.
        text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("model response must be a JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("model response must be a JSON object")
        months = value.get("months")
        rationale = value.get("rationale")
        if type(months) is not int or months < 0 or not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("model response requires nonnegative integer months and rationale")
        return cls(months, rationale.strip())


class Model(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class DemoModel:
    """Deterministic plumbing demo; its output is not a legal prediction."""

    def complete(self, system: str, user: str) -> str:
        return json.dumps({"months": 12, "rationale": "演示输出：未进行法律推理。"}, ensure_ascii=False)


class Bench:
    def __init__(self, presiding: Judge, lay_pool: list[Judge], lay_count: int = 2, seed: int = 0):
        if lay_count < 1 or lay_count > len(lay_pool):
            raise ValueError("lay_count must be between 1 and the size of lay_pool")
        if len({j.name for j in [presiding, *lay_pool]}) != len(lay_pool) + 1:
            raise ValueError("judge names must be unique")
        self.presiding = presiding
        self.lay_pool = lay_pool
        self.lay_count = lay_count
        self.seed = seed

    def select(self, case_id: str) -> list[Judge]:
        rng = random.Random(f"{self.seed}:{case_id}")
        return [self.presiding, *rng.sample(self.lay_pool, self.lay_count)]


def _ask(model: Model, judge: Judge, case: Case, stage: str, prior: dict | None = None, memories: list[dict] | None = None) -> Opinion:
    system = (
        f"你是一名模拟合议庭成员：{judge.role}。视角：{judge.perspective}。"
        "仅根据提供的事实和法条作出研究用途的量刑月数建议。"
        "过往记忆只是参考，不能代替本案事实或法条；若冲突，以本案输入为准。"
        "不要编造法条或案情。只返回 JSON 对象："
        '{"months": 非负整数, "rationale": "简短理由"}。'
    )
    payload = {"stage": stage, "case": asdict(case)}
    if prior is not None:
        payload["prior"] = prior
    if memories:
        payload["memories"] = memories
    return Opinion.from_text(model.complete(system, json.dumps(payload, ensure_ascii=False)))


def run_case(case: Case, bench: Bench, model: Model, rounds: int = 2, memory: MemoryStore | None = None) -> dict:
    """Select, sentence independently, deliberate, and synthesize a final opinion."""
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    judges = bench.select(case.case_id)
    recalled = {judge.name: memory.recall(judge, case) if memory else [] for judge in judges}
    initial = {judge.name: _ask(model, judge, case, "independent_sentencing", memories=recalled[judge.name]) for judge in judges}
    history: list[dict] = []
    current = initial
    for number in range(1, rounds + 1):
        # Every member receives the same previous-round snapshot; no intra-round leakage.
        snapshot = {name: asdict(opinion) for name, opinion in current.items()}
        current = {
            judge.name: _ask(model, judge, case, "deliberation", {"round": number, "opinions": snapshot, "history": history}, recalled[judge.name])
            for judge in judges
        }
        history.append({"round": number, "opinions": {name: asdict(opinion) for name, opinion in current.items()}})
    final = _ask(
        model, bench.presiding, case, "final_decision",
        {"opinions": {name: asdict(opinion) for name, opinion in current.items()}, "history": history}, recalled[bench.presiding.name],
    )
    if memory:
        memory.remember(case, judges, bench.presiding, current, final)
    return {
        "case_id": case.case_id,
        "bench": [asdict(judge) for judge in judges],
        "initial": {name: asdict(opinion) for name, opinion in initial.items()},
        "rounds": history,
        "memory_used": {name: [item["case_id"] for item in items] for name, items in recalled.items()},
        "final": asdict(final),
    }

