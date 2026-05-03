# AGENTS.md — for AI coding agents working ON gpu-summon

> Note: there is also `codeserver/AGENTS.md.template` — that one is rendered
> into the *deployed* code-server box at runtime so agents inside the user's
> IDE know how to use the local LLM. This file you're reading is for agents
> working on the gpu-summon codebase itself.

## What gpu-summon is

A Python launcher (`summon.py`) that rents a vast.ai GPU and serves a local
LLM on it. Two modes:

- **Default** — runs `llama-server` directly in the rented container. opencode
  config gets written locally so the user's laptop talks to the rented LLM
  over the public internet. This is the original gpu-summon.
- **`--with-codeserver`** — adds a browser-based dev environment (code-server)
  on the same rented box, fronted by Caddy with Wildcard-TLS via duckdns +
  Let's Encrypt/ZeroSSL. Custom Docker image `gpu-summon-codeserver` built via
  GitHub Actions, pushed to GHCR. Added in v0.2.0.

User-facing docs: [README.md](README.md). All the painful gotchas we paid
for: [LESSONS.md](LESSONS.md). Read both before changing anything in
`codeserver/` or the launcher's offer/instance flow.

## Most important constraint: vast.ai

Three structural limits we hit and you should treat as load-bearing:

1. **No `--privileged`.** vast.ai's API/SDK exposes no way to launch a
   container with `--privileged` or `--cap-add NET_ADMIN`. Don't propose
   Docker-in-Docker, docker-compose stacks, or anything that needs to
   configure iptables. The original plan was to deploy
   [xaresaicoder](https://github.com/dg1001/xaresaicoder) for per-project
   workspace isolation; we abandoned it for that reason. Native host
   processes (llama-server, code-server, caddy) work fine.

2. **Port 443 is essentially never free on the host.** Mapping `-p 443:443`
   gets us a random host port like `:50241` instead. All user-facing URLs in
   summon.py + AGENTS.md include this random port; the VS Code Ports tab
   strips it (VS Code parser limitation, not fixable from code-server).
   Don't try to "fix" the port-443 retry logic — we removed it on purpose
   because it never succeeds.

3. **TCP/53 outbound to upstream nameservers is firewalled** on most vast
   hosts. Caddy's default DNS-01 propagation self-check trips on this. We
   disabled it via `propagation_timeout -1 + propagation_delay 60s` in the
   Caddyfile snippet `(acme_dns_config)`.

## Roadmap priority: TensorDock

Next backend is **TensorDock**, not RunPod or Lambda. Reasons:

- KVM-VM hosts → real public IP with port 443 free → kills the
  random-host-port mess and the VS Code Ports-tab port-stripping cosmetic.
- `--privileged` works → Docker-in-Docker viable → unblocks bringing
  back full xaresaicoder per-project workspaces as a `--with-xares` mode
  alongside `--with-codeserver`.
- Marketplace pricing model similar to vast.ai → familiar dynamics.
- Has consumer GPUs (RTX 4090) → cost-appropriate for the code-server use
  case (Lambda is H100/A100 only, overkill).

When you take this on, the right shape is a **provider abstraction layer**
in `summon.py` — common CLI surface, pluggable backends. Current
`vastai`-direct calls in `find_best_offers`, `create_instance`,
`wait_until_running`, `cmd_destroy` should live behind a `Provider` ABC.

## Conventions

- **No bcrypt, no per-image basic-auth wrappers.** code-server's built-in
  PASSWORD env handles login. Earlier attempts to wrap with Caddy basicauth
  added complexity for no win.
- **Templates over hardcoded values.** `Caddyfile.template`,
  `AGENTS.md.template`, `xares-env.template` (deleted in v0.2.0 pivot) etc.
  rendered by `envsubst` with whitelisted vars in `onstart.sh`.
- **`set -euo pipefail` in onstart.sh.** Failed steps should be loud —
  silent failures are how we lost an afternoon to GLIBC mismatches.
- **Hot-patchable on running boxes.** Most fixes can be SSHed in and
  applied live without an image rebuild (saves the 10-15min GHA loop). Only
  Dockerfile changes really need a rebuild.
- **Costs money to test.** Each launch is ~$0.10-0.50 depending on how far
  it gets before being destroyed. Use `--dry-run` for iterating on offer-
  search logic; reserve real launches for end-to-end validation.

## Files of note

```
summon.py                      The launcher. Mostly procedural, ~1300 lines.
codeserver/Dockerfile          Image: cuda + llama.cpp + caddy + code-server +
                               opencode + Python/Node dev tools + vim.
codeserver/onstart.sh          Container entrypoint. Must run as set -e
                               and emit clear log lines per phase.
codeserver/Caddyfile.template  Reverse proxy config. envsubst'd at boot.
codeserver/AGENTS.md.template  Rendered into the deployed box for the
                               user's own AI agents.
.github/workflows/build-codeserver-image.yml
                               GHA pushes :latest and :<sha> to GHCR on
                               push to main affecting codeserver/.
LESSONS.md                     The graveyard of debugging hours. Read first.
```

## Don't

- **Don't** propose DinD, docker-compose stacks, or `--privileged` features
  for the vast.ai backend. (TensorDock is fine when we get there.)
- **Don't** silently pin to UID 1000 in Dockerfile (`useradd -u 1000` fails
  because the upstream Ubuntu 24.04 base ships an `ubuntu` user there).
- **Don't** assume `sudo -H -u <user>` actually sets HOME — sudoers'
  `env_keep += HOME` overrides it. Pass `env HOME=...` explicitly.
- **Don't** symlink binaries from `/root/...` to `/usr/local/bin/` —
  `/root` is mode 0700 and other users can't traverse. Copy instead.
- **Don't** add path-based port forwarding (`/proxy/<port>/`) as the
  recommended pattern — it breaks relative URLs in proxied apps. Always
  push users at the dot-subdomain pattern (`<port>.<domain>`).
