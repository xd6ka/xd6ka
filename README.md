# Agents and RAG, measured on a laptop CPU

I build small AI systems and then try to falsify them. Everything in these four
repositories runs on one machine with a local 3B model — no paid APIs, no GPU —
and every claim in them is backed by a report that is committed next to the code.

The through-line: **a working demo and a measured system are different things**, and
the interesting number is usually the failure rate, not the happy path.

| project | what it is | the number worth reading |
|---|---|---|
| [agent-toolbench](https://github.com/xd6ka/agent-toolbench) | tool-calling agent over a sandboxed workspace, with its own 41-task benchmark | 13/37 tasks (0.324–0.351 over three runs); 6/6 refusals at 0-shot, **4/6 with few-shot** |
| [pdf-field-extractor](https://github.com/xd6ka/pdf-field-extractor) | invoice field extraction with schema validation, repair and grounding checks | F1 **0.997** on clean pages, **0.875** on degraded ones — precision stays 1.000 and only recall falls; hallucination 0.167 |
| [rag-service](https://github.com/xd6ka/rag-service) | the RAG pipeline served: queue, two-tier cache, p50/p95 | cached answer **58 ms** after moving the cache ahead of the generation slot (was 7741 ms) |
| [rag-quality-gate](https://github.com/xd6ka/rag-quality-gate) | 60-question golden set over the same pipeline, with a gate that fails when quality drops | four models from 397 MB to 1.9 GB, each failing somewhere else — citation support **1.000** at 3B vs **0.550** at 1.5B and **0.391** at 0.5B; on eight real arXiv papers instead of the fixture, accuracy **0.762** and citation **0.563** |

## What each one actually shows

**agent-toolbench** — six sandboxed tools, one JSON action per step, tool errors fed
back as observations, `step_limit` never faked into an answer. The benchmark grades
the answer, the file state, the refusals *and* whether a tool was used at all. Its
README lists five failure classes taken from traces, including a fake "not found"
written after the file was never opened.

**pdf-field-extractor** — a 60-line PDF writer generates 21 invoices in 7 layout
variants; the pipeline normalises money and dates, validates the schema
(`subtotal + tax == total`), repairs at most two fields, and checks that every value
is literally present in the document. Two findings. **Grounded ≠ correct** — 99.5 % of
returned values are printed in the document, and one of them is still wrong. And a clean
score is a point, not a curve: 12 more documents apply six degradations two at a time,
where rotation and a `PAID` stamp with a competing total turn out to be free, character
spacing costs the labels but not the money, a subtotal printed as `4,062.` + `56` comes
back as `4062.00` and is caught by the arithmetic check, and the one document extracted
perfectly is *withheld* because grounding cannot see a value the layout cut in half.

**rag-service** — FastAPI around the RAG pipeline, with a SQLite job queue and an
explicit `429 + Retry-After` instead of invisible waiting. Three results worth the
repository: a benchmark bug (a new HTTP client per request added 380–680 ms of fake
latency), a placement bug (the cache sat inside the generation slot, so a repeat
question waited 7.7 s behind the model), and an observability bug — `/health` said
`queue_depth: 0` while three requests sat in `acquire()` for forty seconds. Fixing the
third one is what made this measurable: fire six asks for a cached question into a
saturated slot and the placement flag decides whether the first of them waits 73 s or
135 ms, while the fast path itself slows from 24 ms to 398 ms because the lookup embeds
on the same cores as the model. The repo also serves a one-page console at `/demo` that
calls the real endpoints — the screenshot below is that console after one live run, cold
answers 24.8–35.4 s and the same questions served again in 21–30 ms.

<img src="https://raw.githubusercontent.com/xd6ka/rag-service/main/docs/demo/console_run.png" width="460" alt="rag-service live console: four questions, cold vs cached">

**rag-quality-gate** — eight generated pump datasheets, 60 questions with a known gold
string *and* the page that states it, scored three ways that are usually merged: was the
right page retrieved, was the answer correct, was the *cited* block the page with the fact.
Runs of the same pipeline at two recall depths, plus four local models from 397 MB to
1.9 GB. The gate is a script that compares against a committed baseline, derives its
threshold per metric from each metric's denominator, and exits `1` on a regression. Four
findings: the pipeline's default recall depth cut at 12 of the fixture's 24 chunks, so four
questions never showed the reranker the page it would have ranked first — raising it to 24
lifts hit rate and accuracy 0.900 → 1.000 with no model change; the gate's first version
flagged a hallucination *drop* as a regression, because it did not know two of the metrics
are error rates; size is not one axis — `qwen2.5:0.5b` refuses nothing at all (hallucination
1.000), `llama3.2:1b` refuses seven answerable questions yet cites every right answer,
`qwen2.5:1.5b` answers 40/40 and cites only 0.550, and 3B only fails at abstention and
cross-document questions; and multi-hop stays at 5/10 even with twice the context, so that
gap is the model, not the prompt. A fifth measurement is an ablation with no model at all:
two 384-dimension embedders, retrieval only. The challenger wins the dense stage
(0.875 vs 0.750 hit), the cross-encoder erases the difference when the whole fixture fits
in the candidate window, and at the pipeline's shipped 12 of 24 chunks the gap returns
(1.000 vs 0.900) — so the lever is the window, and the embedder would cost a re-index to
buy what one number already buys. The sixth measurement is a different corpus, because the
generated one was grading the pipeline on its own easy cases: eight open-access arXiv papers
on retrieval, 1,005 chunks, 25 questions whose every label is a quoted sentence and the page
it prints on. Accuracy falls 1.000 → 0.762, citation support 1.000 → 0.563, refusal recall
0.700 → 0.500 — and one refusal becomes *"GraphRAG reports an exact match score of 49.4 on
the HotpotQA dev set [3]"*, a number, a benchmark and a citation for a sentence no paper
contains. Building that set also found a scorer bug the fixture could never expose:
`\b57%\b` never matches, so five verbatim-correct percentage answers were scored wrong
(accuracy 0.524 → 0.762 after one regex, no model re-run).

## How the numbers were made

- Local `qwen2.5:3b` through Ollama, CPU only, `temperature` 0 or 0.1, one machine.
- Deterministic fixtures: the corpus is regenerated per run, so a number can be
  reproduced and a failure can be attributed.
- Three runs per configuration where a single run could be noise, and the stability
  of each task across those runs is reported (which tasks always pass, always fail,
  and which flip).
- Reports are committed: `eval/reports/*.json` and `bench/reports/*.json` hold the
  per-task status, the answer, the tools called and the failed check.
- The scoring and sandbox tests run on GitHub Actions on every push across all five
  repositories — the four projects and the base pipeline — `python -m pytest -q`, no
  model, no GPU. The model runs stay local, because the runners have no Ollama: CI
  proves the measuring code works, not that the numbers are big. Everything is MIT
  licensed, because a repository without a licence is a repository nobody may use.
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

Four projects shipped, each with its reports committed, and the measuring side of
`rag-quality-gate` is as far as a laptop goes: 60 labelled questions, a band per metric, a
1.5B-vs-3B comparison, a repeat that reproduced all 60 verdicts, one retrieval finding the
numbers made visible, and a second corpus of eight real papers that halved the accuracy and
found a scorer bug in the process. CI gates the committed reports in both directions — the
improvement must pass, the smaller model must fail — and re-reads every arXiv label against
the PDF it claims before the suite is allowed to pass. What is still missing is scale rather
than machinery: two model sizes, 85 questions across two sets, one machine, no GPU. A decision
about which model to ship wants more of both.
