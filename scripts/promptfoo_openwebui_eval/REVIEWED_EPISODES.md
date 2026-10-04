# Reviewed episodes for evaluation and optimization

DEA measures generated documents and supplies feedback for optimization. An
agentic RAG experiment adds evidence quality and the cost of obtaining an answer.
Keep these dimensions separate: a correct final answer does not establish that
retrieval, grounding or the whole episode was acceptable.

## Repository boundary

| Keep in Git | Keep in private experiment storage |
|---|---|
| Reusable evaluation/export code, synthetic tests, documented schemas and configuration templates | Source documents subject to their licenses, original QA/references, derived datasets |
| Metric definitions and reproducible command examples | Full wire captures, prompts/reasoning, grades, manual reviews, usage/cost ledgers |
| Small deterministic fixtures already used by this bundle | Generated SFT/preference exports, model weights, adapters, caches and logs |
| Explicit implementation gaps below | Local release patches/bundles, Axolotl/RunPod controllers and experimental worktrees |

The [multi-turn pipeline](posttrain/README.md) now includes the reusable Strategy A
recorder, collection, grading, task adapters and SFT/preference construction.
GPU execution and private operator campaign scripts remain external. Keep the
original delivery and experiment archive until independently backed up. Ignoring
a directory is not a backup. Existing benchmark fixtures remain unchanged.

Use an experiment directory outside the checkout, or the ignored `results/`
directory. Keep source attempts immutable and make each export in a new directory.
No move is required for existing experiments: their manifests and capture paths
may depend on the current layout. If relocating later, copy the complete archive,
verify hashes, and update/verify its path references before removing the old copy.
Do not copy only `sft.jsonl`: it loses the evidence and review provenance.

## Offline export

From this bundle directory, using paths to your existing reviewed experiment:

```bash
python scripts/export_reviewed_episodes.py \
  --catalog "$EXPERIMENT_DIR/episodes.jsonl" \
  --cases "$EXPERIMENT_DIR/dataset/cases.jsonl" \
  --output "$EXPERIMENT_DIR/covered-002"
```

This previews counts without writing files. Repeat with `--execute` to create
the export. No network request, judge call, provisioning or training occurs.
Only the standard library is required. An existing output directory is refused.

| Output | Role |
|---|---|
| `episodes.jsonl` | Complete reviewed successes and semantic failures, original fields/usage retained, explicit labels and link to an accepted solution |
| `questions.jsonl` | Original TRAIN cases for the retained questions, including unchanged reference fields |
| `manifest.json` | Counts, excluded question IDs, input/output SHA-256 hashes and source paths; written last as the completion marker |

The export is a private derived view. Source attempts, including incidents and
questions without a solution, remain in the original archive. Capture paths are
preserved relative to the **source catalog**, not rebased or copied. The source
archive must remain available. References in `questions.jsonl` belong to evaluation;
they must never be added to teacher prompts or uploaded retrieval documents.

## Input and selection contract

Inputs are JSONL objects. `--cases` must contain unique `case_id` values and a
`split`; every episode must match a `train` case. Dev/test cases can exist in the
case file but cannot appear in the episode catalog.

Each episode has `case_id`, boolean `accepted`, and one of these statuses:

- `ACCEPTED`: requires a passing automatic grade, passing hard gates and an
  accepting manual review.
- `SEMANTIC_REJECTED`: complete reviewed failure, including a manual veto of an
  automatic pass. It is retained only if the same question has an accepted episode.
- `TECHNICAL_FAILURE` or `BUDGET_INCOMPLETE`: excluded from training negatives.

Complete episodes additionally require a unique `benchmark_id`, `capture_hash`,
`usage` object, `manual_review.accepted`, and `grade` with matching `benchmark_id`
and `capture_hash`, boolean `pass`/`hard_ok`, and
`metrics.correctness.score` / `metrics.groundedness.score` when hard gates passed.
Judge scores must be finite numbers in [0, 1]. Missing/invalid judge results must
be classified as technical failures upstream, never as semantic negatives.
When hard gates failed, exported quality scores are `null`: placeholder zeros
must not be interpreted as measured judgments. All original grade fields survive.

The exporter checks metadata consistency, not the truth of an answer or the
contents of a raw capture. Its hashes establish input/output identity only.
The collector/reviewer must verify actual evidence, the grading receipt and all
LLM requests/responses before declaring an episode complete. A forged or stale
review cannot be detected from these metadata alone.

## Measures and interpretation

`usage` describes the **whole episode**, summing every model call, not only the
final response. The exporter preserves it without estimating missing values.

| Field | Meaning |
|---|---|
| `llm_calls`, `tool_calls` | Model requests and tool calls; one model request can request several tools |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | Reported input/output/total usage across the episode |
| `cached_input_tokens`, `reasoning_tokens` | Subsets of input/output usage; do not add them again to totals |
| `seconds` | Reported episode duration; keep its timing boundary consistent across experiments |
| `quality_scores`, `label` | Judge correctness/grounding scores and final reviewed success/failure |

Missing usage stays unknown, never zero. Supervised training tokens are a separate
measure from generated tokens. Provider charges, tool-internal LLM calls and
indexing/storage costs require their own measured accounting; this export does
not reconstruct them from token counts.

Selection after retries is useful for training but is **not pass@1**, and it biases
comparisons toward solved questions. Compare models on the same fixed evaluation
questions and protocol using all scheduled attempts, including technical failure
rates. Report success counts and uncertainty, quality versus tokens/time/tool
calls, and per-question outcomes. Report selected-training statistics separately.
Question paraphrases/translations from the same source group are not independent
evaluation samples.

## Implemented and remaining validation

- [x] Private wire recorder and native OpenWebUI transport, with multi-turn
  integration tests using simulated transports and reconstruction of real captures.
- [x] ViDoRe and task-CSV adapters, split-bound retrieval, reviewed catalog builder,
  complete success/failure trajectories and paired SFT exports with/without thinking.
- [x] Preference construction restricted to the same observed state; technical
  incidents and manual vetoes cannot become automatic score-margin preferences.
- [ ] Validate each new live model/tool deployment. Test automatic attachment RAG
  and full-document context as separate modes; the collection CLI currently uses
  agentic KB retrieval. The markdown derivative is not official visual ViDoRe.
- [ ] Check actual training tokenizer tokens, labels, masking and context length
  for the selected SFT variant in its external training consumer.
- [x] Build an offline fixed-protocol comparison report with pass@1, retrieval/grounding
  gates, observed cumulative provider usage/cost, reconciled failure rates and
  source-group confidence intervals. Configuration/receipts are frozen and test
  comparison requires explicit unsealing. Hidden-call and whole-episode totals
  stay unknown pending instrumentation; see [comparison contract](posttrain/README.md#fixed-protocol-comparison-report).
- [ ] Instrument any tool-internal LLM calls before claiming complete episode costs
  for summarization tools. The top-level provider recorder cannot observe a tool
  that calls a different endpoint internally.

Run offline regression checks from the repository root:

```bash
python -m pytest tests/test_promptfoo_openwebui_bundle.py tests/test_reviewed_episodes.py -q
```
