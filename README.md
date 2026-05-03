# gpu-summon

One command to summon a self-hosted [llama.cpp](https://github.com/ggml-org/llama.cpp) endpoint on a rented cloud GPU and wire it into your editor. Built for coding sessions where data sovereignty matters — your code never touches a cloud LLM API.

GPU marketplace integration: currently [vast.ai](https://vast.ai). Designed so other providers (RunPod, Lambda, Salad, …) can be added without changing the launcher's UX.

> Languages: **English** · [Deutsch](README.de.md) · [LESSONS.md](LESSONS.md) (deep dive)

## What this does

- Rents a cloud GPU and runs `llama-server` on it with any GGUF model (default: Unsloth's Qwen3.6-27B Q5_K_XL)
- Picks the offer for you using cost + reliability + bandwidth + CUDA-version filters
- Optionally **smoke-tests the actual download bandwidth** before committing to the host (rented GPUs often advertise 200 Mbps and deliver 80)
- Generates a Bearer-token API key by default — your endpoint is **not** public out of the box
- Writes a ready-to-go [opencode](https://opencode.ai) config so you can just type `opencode` and start coding
- One-command teardown so you don't forget the meter running

## Why?

Cloud-hosted coding LLMs are expensive and your code is yours. For sessions where data locality matters but a 32 GB local GPU isn't in the cards, renting a 30 ¢/h cloud GPU and running a self-hosted llama-server is the sweet spot — and `gpu-summon` exists to compress 30 minutes of CLI archaeology into one command.

## Quick start

```bash
git clone https://github.com/<you>/gpu-summon.git
cd gpu-summon
pip install vastai requests

export VAST_API_KEY=...   # from https://cloud.vast.ai/manage-keys/

# Solo coding session: 64k context per user, secure-by-default
python summon.py --solo-mode

# When done
python summon.py --list
python summon.py --destroy <id>
```

The launch prints the endpoint URL + API key and writes `~/.config/opencode/opencode.json` + `~/.local/share/opencode/auth.json`. Run `opencode` and you're chatting with your own GPU.

## Features

- One-shot launch: search → create → wait → write config → done
- Three operational modes:
  - default — 4 slots × 16 k context (small team)
  - `--workshop-mode` — 8 slots × 16 k context (5–10 concurrent users)
  - `--solo-mode` — 2 slots × 64 k context (single user, long sessions)
- **Real bandwidth smoketest** — measure HF→host throughput before the long wait, automatically retry the next offer if slow
- Auto-generated 32-char Bearer token (override or disable explicitly)
- Cost guards: `--max-price`, `--max-inet-cost`, `--min-inet-down`, `--min-reliability`
- Vast.ai Template support (`--template-hash`) for reusable launch configs
- Diagnostic mode (`--debug`) showing exactly which filter eliminates results
- Convenience: `--list`, `--destroy ID`, `--dry-run`

## Requirements

- Python ≥ 3.10
- `vastai` SDK + `requests` (`pip install vastai requests`)
- A vast.ai account with API key in `VAST_API_KEY` (current GPU marketplace; see [Roadmap](#roadmap))
- For the bandwidth smoketest: `ssh-keygen` + `ssh` (any Linux / macOS box)

## Modes

| Flag | parallel | ctx (total) | per slot | Best for |
|---|---|---|---|---|
| (default) | 4 | 65 536 | 16 384 | small team, mid-length chats |
| `--workshop-mode` | 8 | 131 072 | 16 384 | 5–10 concurrent users, demo |
| `--solo-mode` | 2 | 131 072 | 65 536 | single user, long context |

Override with `--parallel N --num-ctx M` for custom setups. For 32 GB cards the practical ceiling with q8 KV-cache is ~256 k total context.

## Bandwidth smoketest

Rented GPU hosts advertise a network speed at offer time, but the *real* HuggingFace download bandwidth varies wildly by route, time of day, and HF mirror state. We've measured 88 Mbps real on a "200 Mbps advertised" host (more than 50 % shortfall is common).

`--min-real-mbps N` enables a real measurement:

1. Container starts, `llama-server` begins downloading the GGUF
2. Launcher SSHes in (ed25519 key auto-generated to `~/.ssh/gpu_summon_smoketest`, attached via the marketplace API)
3. Polls until `.downloadInProgress` cache file appears (max 90 s)
4. Samples file growth for 15 s → real Mbps
5. If below threshold: destroy and try next offer from the top-5 list
6. After 5 misses: exit with a hint

Cost: ~$0.013 per failed candidate. Worst case (all 5 reject): ~$0.06 + 8 minutes wait.

```bash
# Strict — only hosts that actually deliver 200+ Mbps
python summon.py --solo-mode --min-inet-down 500 --min-real-mbps 200

# Loose — 100 Mbps is enough
python summon.py --solo-mode --min-real-mbps 100
```

`--min-inet-down` filters by *advertised* bandwidth (cheap, no instance created); `--min-real-mbps` validates the *actual* bandwidth (creates instance, may destroy + retry).

## Authentication

By default the launcher generates a 32-char Bearer token and passes it as `--api-key` to llama-server. Without that, your endpoint is a publicly reachable inference target on someone else's GPU bill.

```bash
# Default: auto-generate
python summon.py --solo-mode
# [auth]   Kein --llm-api-key angegeben - generiert: ZeFiOs...

# Reuse a fixed key across launches
GPU_SUMMON_LLM_API_KEY=mykey python summon.py --solo-mode

# Explicitly open (only inside trusted networks!)
python summon.py --solo-mode --llm-api-key ''
```

The opencode auth file (`~/.local/share/opencode/auth.json`) is written with the same token.

Note: `/v1/models` is *not* protected by llama.cpp's auth — only inference endpoints (`/v1/chat/completions`, `/completion`) are. That's upstream behavior, not a bug here.

## opencode integration

Two files are written by `--write-config` (default on):

- `~/.config/opencode/opencode.json` — provider definition (baseURL, model id, tools enabled)
- `~/.local/share/opencode/auth.json` — Bearer token

Just run `opencode` after launch finishes. The provider id is `vast-llamacpp`, the model id is auto-detected from `/v1/models`.

## Cost ballpark

Numbers from vast.ai (April 2026); other providers have similar order of magnitude.

| What | Typical |
|---|---|
| Hourly | $0.30–0.40/h on RTX 5090 |
| Model download (~21 GB UD-Q5_K_XL) | $0.02–0.10 one-time |
| 1 h interactive coding | ~$0.40 |
| 8 h workshop session | ~$3 |
| Smoketest probe (per failed candidate) | ~$0.013 |

`--max-price 0.45 --max-inet-cost 0.01` enforces these as hard filters — over-budget offers never get considered.

## CLI reference

```
--backend {llamacpp,ollama}    inference engine (default: llamacpp)
--model SPEC                   model spec, e.g. unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL
--min-vram N                   min GPU VRAM in GB (default: 32)
--max-price F                  max $/h (default: 0.50)
--min-reliability F            min reliability (default: 0.95)
--min-inet-down N              min advertised Mbps (default: 200)
--max-inet-cost F              max bandwidth cost in $/GB (default: 0.01)
--min-real-mbps F              real-bandwidth smoketest threshold; 0 disables (default: 0)
--cuda-min F                   min CUDA version (default: 12.0; 13.2 always blocked)
--region CODE                  geo filter, e.g. EU, US (default: any)
--disk N                       disk in GB (default: 80)

--num-ctx N                    total context window (default: 65 536)
--parallel N                   parallel slots (default: 4)
--workshop-mode                shortcut: parallel=8, ctx=131072
--solo-mode                    shortcut: parallel=2, ctx=131072

--llm-api-key TOKEN            Bearer token; default auto-generated; '' disables auth
--template-hash HASH           use a vast.ai Template instead of self-rendered onstart

--label NAME                   instance label (default: opencode-llm)
--timeout N                    container-up timeout (default: 600 s)
--model-timeout N              model-ready timeout (default: 1800 s)
--write-config                 write opencode config (default on)
--dry-run                      print plan, don't create

--list                         list active instances
--destroy ID                   destroy instance
--debug                        diagnose offer-search filter chain
```

Environment variables:

- `VAST_API_KEY` — vast.ai API key (required)
- `GPU_SUMMON_LLM_API_KEY` — reusable Bearer token for the endpoint
- `GPU_SUMMON_TEMPLATE_HASH` — pinned vast.ai Template hash

## Files written

- `~/.config/opencode/opencode.json` — opencode provider config
- `~/.local/share/opencode/auth.json` — opencode auth tokens
- `~/.ssh/gpu_summon_smoketest{,.pub}` — auto-generated, only if `--min-real-mbps > 0`

## Vast.ai Templates (advanced)

Vast.ai-specific. You can capture a working configuration as a Template and reuse it across launches:

```bash
GPU_SUMMON_TEMPLATE_HASH=<hash> python summon.py --solo-mode
```

The launcher uses the template's image + onstart, only injecting mode-specific env (`LLAMA_PARALLEL`, `LLAMA_CTX`, `LLAMA_MODEL`, `LLAMA_API_KEY`). See [LESSONS.md](LESSONS.md#vastai-templates) for how to create one.

## Troubleshooting

**"Keine passenden Offers gefunden"** — relax filters: `--max-price 0.50 --min-reliability 0.92`, drop `--region`. Use `--debug` to see exactly which filter zeroes the result count.

**Endpoint hangs at HTTP 000 for 10+ min** — model is still downloading. llama.cpp's `-hf` flag is silent; SSH in and `stat -c%s` the `.downloadInProgress` file to monitor progress. Use `--min-real-mbps` to bail early on slow hosts.

**Container crashes with `libllama-common.so.0` not found** — happens when manually restarting llama-server inside the container. Set `LD_LIBRARY_PATH=/app` and `HF_HOME=/workspace/hf_cache`.

**`Bind for :::PORT failed: port is already allocated`** — bad host. Destroy + retry; the offer pool is large enough that one bad host is no problem.

For deeper diagnostics, see [LESSONS.md](LESSONS.md).

## code-server Mode (`--with-codeserver`)

The same rented machine can host **the LLM and a browser-based VS Code** ([code-server](https://github.com/coder/code-server)) on one box, fronted by Caddy with Wildcard-TLS via duckdns + Let's Encrypt. One URL, no local install, full TLS, password-protected via code-server's built-in login.

```bash
# One-time setup at https://www.duckdns.org/ (free, no credit card):
#   1. Sign in (GitHub/Google/Twitter/Reddit OAuth)
#   2. Pick a free subdomain — the form tells you immediately if it's taken
#   3. Copy your account token from the top of the page
# Use *your* chosen name everywhere `mybox` appears below — it's just an example.
export DUCKDNS_TOKEN=...
export VAST_API_KEY=...

# Launch llama-server + code-server + Caddy on a single rented GPU
python summon.py --with-codeserver --code-domain mybox --solo-mode
# → Browser-IDE:   https://mybox.duckdns.org      (login: <printed password>)
# → LLM-Endpoint:  https://llm.mybox.duckdns.org/v1   (Bearer: <printed key>)

# Cleanup (destroys instance AND clears the duckdns A-record)
python summon.py --destroy <id> --duckdns-token $DUCKDNS_TOKEN
```

If the token doesn't match the subdomain (typo or never registered), the first DNS update returns `KO` and Caddy's cert request fails — register the subdomain first, then launch.

**What's running on the box:**

```
┌─ vast.ai container (no Docker, all native) ──┐
│  llama-server (host process, GPU)            │
│  code-server (host process, password auth)   │
│  Caddy :443 → TLS, routing                   │
└──────────────────────────────────────────────┘
```

We originally tried Docker-in-Docker for full per-project workspace isolation but vast.ai disallows `--privileged` on shared hosts — `dockerd` can't configure iptables and dies at boot. See [LESSONS.md § code-server Mode](LESSONS.md#code-server-mode-with-codeserver) for the full story.

**Requirements beyond the standard mode:**

- A duckdns.org subdomain + token (free)
- The custom image `ghcr.io/dg1001/gpu-summon-codeserver:latest` (built once via `codeserver/build-and-push.sh` or the GitHub Actions workflow)
- `--disk` is auto-bumped to 60 GB for image + model cache + IDE state

## Roadmap

- [ ] Pluggable GPU marketplace backends (RunPod, Lambda, Salad)
- [ ] Optional vLLM / SGLang backends alongside llama.cpp
- [ ] Native CLI entry point (`gpu-summon` instead of `python summon.py`)
- [x] Optional reverse proxy with TLS termination (`--with-codeserver` ships with Caddy + duckdns wildcard cert)

## Contributing

PRs welcome. Please:

- Test with `--dry-run` before real launches
- Document non-obvious findings in [LESSONS.md](LESSONS.md)
- Keep changes scoped — this is a single-file launcher, not a framework

## License

MIT — see [LICENSE](LICENSE).
