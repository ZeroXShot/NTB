# 11. A training run gets its own process

**Status:** Accepted · 2026-08-24

## Context

The studio is a long-lived local server holding the document a person is editing.
Training is the opposite kind of work: it allocates most of the machine's memory,
blocks for minutes at a time, and fails in ways Python does not always survive —
a CUDA out-of-memory, a segfault in a native kernel, a `sys.exit` from somebody's
data script. Any of those inside the server takes the editor down with the run,
and loses the document that was being edited.

Running it in a thread is not an answer either: a thread cannot be killed, and
"stop this run" has to actually stop it.

## Decision

A run is a subprocess. `ntb.runs.worker` is a module with a `__main__`, launched
with a JSON config file, and everything it has to say it says on stdout as one
JSON object per line. The manager reads that stream on a thread, records it in
SQLite, and forwards it to whoever is listening.

Stopping a run terminates the process. Metrics live in a database rather than in
memory, so a run outlives the session that started it, and the session that
finds it later can read what happened.

## Consequences

* An out-of-memory kills the run, not the studio, and the studio can say so.
* Stop works, because a process can be terminated.
* Everything crossing the boundary has to be serialisable, which keeps the
  protocol small and makes the same events usable by the CLI, the studio, and
  the MCP server later.
* A run costs a process start and an interpreter's worth of memory. For a
  training run that is noise.
* The worker loads the document *from disk*, so the studio saves before it
  starts one. Training a file that does not match the screen would be worse.
* A run whose session ended is marked stopped when a manager next opens the
  store: its process went with the session, and pretending otherwise would
  leave runs "running" forever.
