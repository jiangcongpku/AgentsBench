# AgentsBench: paper workflow implementation

This repository contains the implementation of the multi-agent simulation framework in Jiang and Yang (2025)(https://doi.org/10.3390/systems13080641): bench selection, independent sentencing, multi-round deliberation, and final decision. 

**Scope:** The framework is for research, not legal advice or actual sentencing.

## Quick start

Python 3.10 or later is required. No third-party package is needed for the demo.

```bash
python -m pip install -e .
agentsbench --provider demo --cases examples/cases.jsonl --output outputs/results.jsonl
```

The sample case is fictional. The output records the selected bench, each member's independent opinion, every deliberation round, and the presiding judge's final opinion. `outputs/` is ignored by Git.

## Cross-case memory

Memory is enabled by default. After a case completes, the program saves each selected member's opinion in `outputs/<provider>-memory.json`. The presiding judge retains the final legal rationale; lay members retain their own last-round social observations. When a later case has the same charge or identical statute text, each member receives up to three of **their own** relevant memories in the model prompt. Members never receive another member's stored memories. The output's `memory_used` field lists the recalled case IDs, making this behavior inspectable. Memory is committed only after a case finishes successfully and is replaced when the same case ID is rerun.

Use `--memory path/to/memory.json` to choose a store or `--no-memory` to run without it. To start a fresh experiment, point `--memory` at a new path. Memory files may contain case-derived reasoning and are ignored when stored under `outputs/`; keep other memory paths private and do not commit them to a public repository. This is a simple case-retrieval memory, not model training or a verified legal precedent database.

For an OpenAI-compatible chat-completions endpoint, set `OPENAI_API_KEY`, optionally `OPENAI_MODEL` and `OPENAI_BASE_URL`, then run:

```bash
agentsbench --provider api --cases examples/cases.jsonl --rounds 2
```

Do not put credentials in a configuration file committed to Git. The `.env.example` file contains placeholders only; this program reads environment variables directly and does not load `.env` automatically. API calls may send case facts to the configured provider, so use only data you are authorized to transmit. The program makes one request per opinion and does not provide retry or budget controls.

## Input and output

Input is UTF-8 JSONL, one case per line:

```json
{"case_id":"case-001","facts":"Case facts here","statutes":"Verified applicable statutes here","charge":"Optional charge"}
```

`case_id`, `facts`, and `statutes` are required nonempty strings. Output is UTF-8 JSONL. Bench selection is deterministic for a given `--seed` and case ID; the presiding judge is fixed and two lay members are sampled from three configured perspectives. `--rounds` controls the number of discussion rounds. Every member sees the previous round's full set of opinions, then updates independently. The presiding judge synthesizes the final opinion.

The role profiles in `src/agentsbench/cli.py` are illustrative and should be adapted to a documented study design. The model must return a JSON object with a nonnegative integer `months` and a nonempty `rationale`; invalid responses stop the run with an error. The result file is opened before processing, so reruns overwrite it.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover the four-stage result shape, deterministic selection, and malformed responses. They do not validate legal accuracy or reproduce published empirical results.

## Paper citation

Cong Jiang and Xiaolei Yang. “AgentsBench: A Multi-Agent LLM Simulation Framework for Legal Judgment Prediction.” *Systems* 13, no. 8 (2025): 641. https://doi.org/10.3390/systems13080641

## Repository layout

- `src/agentsbench/core.py` — case and opinion validation, bench selection, and deliberation workflow.
- `src/agentsbench/providers.py` — optional OpenAI-compatible API adapter.
- `src/agentsbench/memory.py` — persistent, judge-specific case memory and retrieval.
- `src/agentsbench/cli.py` — JSONL command-line runner.
- `examples/cases.jsonl` — fictional plumbing example.
- `tests/` — local checks.

The repository intentionally does not include research datasets, API keys, generated transcripts, private case material, or an open-source license. Add a license only after confirming who owns the rights to this newly written software and what terms they choose.
