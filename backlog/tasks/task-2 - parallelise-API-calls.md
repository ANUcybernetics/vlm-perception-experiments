---
id: TASK-2
title: parallelise API calls
status: Done
assignee: []
created_date: '2026-03-26 22:28'
updated_date: '2026-03-27 03:27'
labels: []
dependencies: []
---

The AI platform providers (OpenAI and Anthropic) do have some "max concurrent
request" rate limiting, but apart from that all of these experiments can be
performed independently.

Using modern python (I'm happy to use even 3.14 if it gives us cleaner and nicer
async/parallel primitives) can we update the scripting interface to do as many
runs as possible in parallel, while respecting each provider's rate limits?

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added async evaluation using AsyncAnthropic/AsyncOpenAI with per-provider asyncio.Semaphore rate limiting. CLI accepts multiple --model and --prompt options with concurrent execution via asyncio.gather. --concurrency flag controls max requests per provider (default 10). --resume flag skips already-completed trials.
<!-- SECTION:NOTES:END -->
