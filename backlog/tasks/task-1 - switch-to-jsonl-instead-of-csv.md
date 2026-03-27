---
id: TASK-1
title: switch to jsonl instead of csv
status: Done
assignee: []
created_date: '2026-03-26 22:28'
updated_date: '2026-03-27 03:27'
labels: []
dependencies: []
---

Would using jsonl be more robust than csv? Also, will the current "write to
file, then commit results" be robust in the presence of multiple simultaneous
experiments (e.g. running different models)?

Also see task-2: if doing the parallelisation in python (rather than in calling
the python script) is a better solution then we can do that instead.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Switched storage from CSV to JSONL format. Results file is now results/results.jsonl using polars read_ndjson/write_ndjson. Existing CSV data converted. Async-aware append with asyncio.Lock for concurrent writes.
<!-- SECTION:NOTES:END -->
