# Round 10: Ling confirmation, panels and whole-round profiling

Started 2026-10-18 07:00 UTC; deadline 11:00 UTC. Branch `l5-cache`. No system changes, live gateway changes, pushes or PRs. The owner subsequently authorized CPU sampling and sudo explicitly; use those for task 3 without changing kernel settings.

## Measurement protocol

The unchanged stage-B foreign-process gate is reused. A rejected request invalidates its whole prompt pass; restart the server before retrying, so the adaptive cache always sees the same warmup and eight-prompt history. Screening: MTP3 control and seven DSpark settings, four code and four English prompts per setting. Confirmation: best two candidates and control, three rotating rounds. Raw output hashes, draft counts, accepted lengths, CPU cost and exact configurations are archived.
