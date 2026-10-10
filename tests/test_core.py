import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentsbench.core import Bench, Case, DemoModel, Judge, Opinion, run_case
from agentsbench.memory import MemoryStore


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.bench = Bench(
            Judge("chair", "professional", "law"),
            [Judge("lay1", "lay", "social"), Judge("lay2", "lay", "family"), Judge("lay3", "lay", "community")],
            seed=7,
        )
        self.case = Case("example", "synthetic facts", "example statute")

    def test_four_stages_and_stable_selection(self):
        result = run_case(self.case, self.bench, DemoModel(), rounds=2)
        self.assertEqual(len(result["bench"]), 3)
        self.assertEqual(len(result["initial"]), 3)
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["final"]["months"], 12)
        self.assertEqual(self.bench.select("example"), self.bench.select("example"))
        json.dumps(result)

    def test_invalid_model_output_rejected(self):
        for text in ('{"months": -1, "rationale": "x"}', '{"months": true, "rationale": "x"}', 'oops'):
            with self.assertRaises(ValueError):
                Opinion.from_text(text)

    def test_tagged_reasoning_is_not_saved_as_opinion(self):
        opinion = Opinion.from_text('<think>internal text</think>\n```json\n{"months": 24, "rationale": "brief reason"}\n```')
        self.assertEqual(opinion, Opinion(24, "brief reason"))

    def test_missing_case_fields_rejected(self):
        with self.assertRaises(ValueError):
            Case.from_dict({"case_id": "x", "facts": "y"})

    def test_memory_persists_and_is_scoped_to_each_judge(self):
        class RecordingModel:
            def __init__(self):
                self.inputs = []

            def complete(self, system, user):
                self.inputs.append(json.loads(user))
                return json.dumps({"months": 18, "rationale": "case-specific observation"})

        bench = Bench(Judge("chair", "professional", "law"), [Judge("lay", "lay", "social")], lay_count=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            store = MemoryStore(path)
            model = RecordingModel()
            first = Case("first", "fictional fact one", "statute A", "charge A")
            second = Case("second", "fictional fact two", "statute A", "charge A")
            run_case(first, bench, model, rounds=0, memory=store)
            self.assertTrue(path.exists())
            self.assertEqual(len(MemoryStore(path).records), 2)
            model.inputs.clear()
            result = run_case(second, bench, model, rounds=0, memory=MemoryStore(path))
            self.assertEqual(result["memory_used"], {"chair": ["first"], "lay": ["first"]})
            self.assertEqual([item["memories"][0]["kind"] for item in model.inputs[:2]], ["legal_reasoning", "social_observation"])
            self.assertTrue(all(len(item.get("memories", [])) == 1 for item in model.inputs))
            self.assertEqual(len(MemoryStore(path).records), 4)
            run_case(second, bench, model, rounds=0, memory=MemoryStore(path))
            self.assertEqual(len(MemoryStore(path).records), 4)

    def test_memory_ignores_unrelated_cases(self):
        bench = Bench(Judge("chair", "professional", "law"), [Judge("lay", "lay", "social")], lay_count=1)
        with TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.json")
            run_case(Case("one", "facts", "article 12", "theft"), bench, DemoModel(), rounds=0, memory=store)
            self.assertEqual(store.recall(bench.presiding, Case("two", "other facts", "section 99", "fraud")), [])


if __name__ == "__main__":
    unittest.main()

