---
id: TASK-2
title: parallelise API calls
status: To Do
assignee: []
created_date: "2026-03-26 22:28"
labels: []
dependencies: []
---

The AI platform providers (OpenAI and Anthropic) do have some "max concurrent
request" rate limiting, but apart from that all of these experiments can be
performed independently.

Using modern python (I'm happy to use even 3.14 if it gives us cleaner and nicer
async/parallel primitives) can we update the scripting interface to do as many
runs as possible in parallel, while respecting each provider's rate limits?
