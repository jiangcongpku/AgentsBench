"""Command-line entry point for JSONL case runs."""

import argparse
import json
from pathlib import Path

from .core import Bench, Case, DemoModel, Judge, run_case
from .memory import MemoryStore
from .providers import MiniMaxModel, OpenAICompatibleModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgentsBench paper workflow on JSONL cases")
    parser.add_argument("--cases", type=Path, default=Path("examples/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    parser.add_argument("--provider", choices=["demo", "api", "minimax"], default="demo")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory", type=Path, help="Persistent judge memories (defaults to a separate file for each provider)")
    parser.add_argument("--no-memory", action="store_true", help="Disable cross-case memory")
    args = parser.parse_args()
    model = {"demo": DemoModel, "api": OpenAICompatibleModel, "minimax": MiniMaxModel}[args.provider]()
    memory_path = args.memory or Path(f"outputs/{args.provider}-memory.json")
    memory = None if args.no_memory else MemoryStore(memory_path)
    bench = Bench(
        Judge("presiding", "审判长", "重视法律依据、事实认定和说理"),
        [
            Judge("lay_education", "人民陪审员", "关注教育背景与社会影响"),
            Judge("lay_health", "人民陪审员", "关注健康与家庭影响"),
            Judge("lay_community", "人民陪审员", "关注社区视角与公平感受"),
        ],
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.cases.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                case = Case.from_dict(json.loads(line))
                result = run_case(case, bench, model, rounds=args.rounds, memory=memory)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{args.cases}:{line_number}: {exc}") from exc
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

