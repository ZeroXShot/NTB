# 10. The studio server answers only same-origin requests

**Status:** Accepted · 2026-08-24

## Context

`ntb studio` starts an HTTP server on loopback that can read and write `.ntb`
files anywhere the user can. Loopback is not a security boundary: any page in
the user's browser can POST to `http://127.0.0.1:8756`, and WebSockets are not
covered by the same-origin policy at all.

## Decision

Every HTTP request and every WebSocket upgrade is checked: requests carrying an
`Origin` from anywhere but the server itself or a loopback host are refused.
Requests with no `Origin` at all (curl, the test client, a future MCP client)
are allowed, since those are not browsers acting on someone else's behalf.

The server binds `127.0.0.1` by default. `--host` can widen that, and doing so
is the user's decision, not the default.

## Consequences

* A page on the internet cannot drive the studio or read the user's models.
* A frontend served from the Vite dev server on `localhost:5173` still works,
  which is what makes development possible.
* This is not authentication. Anything already running as the user can talk to
  the server, which is the same trust level as the CLI.
