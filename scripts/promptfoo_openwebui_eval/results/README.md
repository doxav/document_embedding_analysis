# Evaluation results

`results/` contains local generated artifacts and is ignored except for this index. Raw outputs can include large model responses, so only reusable analysis belongs in Git.

## Current validated run

[`tool_update_smoke_20260902_7ae59c`](tool_update_smoke_20260902_7ae59c/README.md) vérifie la version modifiée de `thematic_summary_v5b.py` sur le même premier exemple MultiLexSum avec DTCRS, RAPTOR et Kohaku lancés réellement en parallèle. Le rapport contient les scores, temps, tokens, commandes, traces et sorties Markdown.

[`multilexsum_algorithm_benchmark_20260901_191605`](multilexsum_algorithm_benchmark_20260901_191605/README.md) contains the current first-three-example MultiLexSum algorithm benchmark. RAPTOR, fixed DTCRS, and Kohaku now have 9/9 direct generations and Promptfoo scores, with commands, per-run results, generated Markdown, and cross-algorithm comparisons.

[`bigsurvey_thematic_20260901_180831`](bigsurvey_thematic_20260901_180831/README.md) compares `target_length=long` outside Promptfoo with `target_length=short` through Promptfoo on BigSurvey task `11197428`, then validates direct generation and offline scoring on MultiLexSum task `CJ-AL-0007`. All use OpenWebUI v0.10.2, model `dea-large-thematic-dtcrs`, tool `large_thematic_summarizer`, Valve `algorithm=dtcrs`, and structure `thematic`.

The run proves both orchestration paths work and records quality, output size, OpenWebUI-reported tokens/cost, wall time, raw responses, and generated Markdown.

## Earlier results

- `bigsurvey_first3_20260612_203849/ANALYSIS.md` documents the earlier Step 1-only run, its DEA target-normalization fix, and why Step 2/3 did not complete at that time.
- `current/matrix_summary.tsv` is the historical process/model matrix summary.
- `trace_experiments/` contains optimization/calibration artifacts; those are separate from the current non-optimization benchmark.

The historical model labels are setup-specific. Do not compare their scores with the current run unless dataset rows, generated candidates, judge model, and DEA embedding settings are identical.
