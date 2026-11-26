# Agents and RAG, measured on a laptop CPU

I build small AI systems and then try to falsify them. Everything in these three
repositories runs on one machine with a local 3B model — no paid APIs, no GPU —
and every claim in them is backed by a report that is committed next to the code.

The through-line: **a working demo and a measured system are different things**, and
the interesting number is usually the failure rate, not the happy path.

| project | what it is | the number worth reading |
|---|---|---|
| [agent-toolbench](https://github.com/xd6ka/agent-toolbench) | tool-calling agent over a sandboxed workspace, with its own 41-task benchmark | 13/37 tasks (0.324–0.351 over three runs); 6/6 refusals at 0-shot, **4/6 with few-shot** |
| [pdf-field-extractor](https://github.com/xd6ka/pdf-field-extractor) | invoice field extraction with schema validation, repair and grounding checks | field-level F1 **0.997**, 20/21 documents fully correct, hallucination 0.167 |
| [rag-service](https://github.com/xd6ka/rag-service) | the RAG pipeline served: queue, two-tier cache, p50/p95 | cached answer **58 ms** after moving the cache ahead of the generation slot (was 7741 ms) |

## What each one actually shows

**agent-toolbench** — six sandboxed tools, one JSON action per step, tool errors fed
back as observations, `step_limit` never faked into an answer. The benchmark grades
the answer, the file state, the refusals *and* whether a tool was used at all. Its
README lists five failure classes taken from traces, including a fake "not found"
written after the file was never opened.

**pdf-field-extractor** — a 60-line PDF writer generates 21 invoices in 7 layout
variants; the pipeline normalises money and dates, validates the schema
(`subtotal + tax == total`), repairs at most two fields, and checks that every value
is literally present in the document. The finding: **grounded ≠ correct** — 99.5 % of
returned values are printed in the document, and one of them is still wrong.

**rag-service** — FastAPI around the RAG pipeline, with a SQLite job queue and an
explicit `429 + Retry-After` instead of invisible waiting. Two results worth the
repository: a benchmark bug (a new HTTP client per request added 380–680 ms of fake
latency) and a placement bug (the cache sat inside the generation slot, so a repeat
question waited 7.7 s behind the model). The repo also serves a one-page console at
`/demo` that calls the real endpoints — the screenshot below is that console after one
live run, cold answers 24.8–35.4 s and the same questions served again in 21–30 ms.

<img src="https://raw.githubusercontent.com/xd6ka/rag-service/main/docs/demo/console_run.png" width="460" alt="rag-service live console: four questions, cold vs cached">

## How the numbers were made

- Local `qwen2.5:3b` through Ollama, CPU only, `temperature` 0 or 0.1, one machine.
- Deterministic fixtures: the corpus is regenerated per run, so a number can be
  reproduced and a failure can be attributed.
- Three runs per configuration where a single run could be noise, and the stability
  of each task across those runs is reported (which tasks always pass, always fail,
  and which flip).
- Reports are committed: `eval/reports/*.json` and `bench/reports/*.json` hold the
  per-task status, the answer, the tools called and the failed check.
- Where something was not measured — 7B models, GPU, multi-worker serving — the README
  says so instead of estimating.

## Reproduce any of it

```bash
git clone https://github.com/xd6ka/agent-toolbench && cd agent-toolbench
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                  # 31 tests
ollama serve && ollama pull qwen2.5:3b     # in another terminal
python eval\run_bench.py --shots 0 --out eval\reports\bench_0shot_1.json
```

A full agent run takes 9–18 minutes on this machine; the extractor and the service
benchmarks are quicker and documented in their own READMEs.

## Currently

Adding the layer all three are missing: **answer quality for RAG** — retrieval
hit-rate and MRR, plus whether an answer is supported by the passage it cites.
"Retrieved the right chunk" and "answered correctly from it" are different properties,
and only one of them is easy to measure.
