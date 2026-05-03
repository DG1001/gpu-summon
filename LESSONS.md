# LESSONS.md

Things we've learned the hard way running self-hosted Qwen3.6-27B on rented cloud GPUs. Most of these are not in any official docs.

> Note: the current implementation of `gpu-summon` uses **vast.ai** as the GPU marketplace, so the API-quirk and SSH-recipe sections are vast-specific. The bandwidth, llama.cpp, VRAM, performance, and cost sections generalize to any rented cloud GPU.
>
> Sister docs: [README.md](README.md) (English) · [README.de.md](README.de.md) (Deutsch)

## Contents

- [Bandwidth: advertised vs real](#bandwidth-advertised-vs-real)
- [vast.ai API quirks](#vastai-api-quirks)
- [llama.cpp gotchas](#llamacpp-gotchas)
- [Silent model download](#silent-model-download)
- [Performance](#performance)
- [VRAM math](#vram-math)
- [SSH debug recipes](#ssh-debug-recipes)
- [Manual llama-server restart](#manual-llama-server-restart)
- [Cost reference](#cost-reference)
- [vast.ai Templates](#vastai-templates)
- [code-server Mode (--with-codeserver)](#code-server-mode-with-codeserver)

---

## Bandwidth: advertised vs real

Vast hosts advertise a `inet_down` Mbps figure on each offer. **It lies, frequently, by 50% or more.** A "200 Mbps" host can deliver 80 Mbps real to HuggingFace.

We measured this concretely:

| Run | inet_down advertised | Real HF download | Shortfall |
|---|---|---|---|
| First 5090 host | 220 Mbps | ~50 MB/s ≈ 400 Mbps | (over-delivered, rare) |
| Later 5090 host | 200 Mbps | 11 MB/s ≈ 88 Mbps | -56 % |

Causes (best guess): HF mirror state, route to closest CDN edge, ISP policy on the host, time-of-day congestion. Re-renting the same host an hour later can flip the result.

**`--min-real-mbps` mitigates this**: launcher SSHes in after `wait_until_running`, samples the cache file's growth rate for 15 s, retries the next offer if too slow. See [README.md § Bandwidth smoketest](README.md#bandwidth-smoketest).

The smoketest is more honest than a Cloudflare speedtest because it measures the *actual* HF→host throughput, not idealized WAN bandwidth.

---

## vast.ai API quirks

### `gpu_ram` query is GB, response is MB

```
gpu_ram>=29760    # interpreted as 29 760 GB → 0 hits
gpu_ram>=29.76    # correct (32 GB cards with 7 % reserve buffer)
```

Floats are allowed in the query. Response gives MB (32607 for 5090, 97887 for 6000 Ada).

### `reliability` (query) vs `reliability2` (response)

Vast-internal inconsistency. In queries: `reliability>=0.97`. In responses: `reliability2` is the active field.

### `inet_down_cost` is `$/GB`

Not `$/MB`, not `$/TB`. The launcher caps it at 0.01 by default — over 80 % of offers fall under that. Above 0.05 a 21 GB pull becomes noticeable ($1+).

### `cuda_max_good` lexicographic comparison (sometimes)

The vast query engine compares version strings lexically in some cases, so `cuda_max_good >= '12.10'` falsely matches `'12.4'` (because `'12.10' < '12.4'` as strings). The launcher therefore filters CUDA **Python-side** with float comparison.

### CUDA 13.2 produces gibberish on Qwen3.6

Hard-coded as `CUDA_BLACKLIST = ["13.2"]`. Known driver bug — random tokens instead of content. The launcher excludes 13.2 for all backends.

### `create_instance` SDK does not accept `ssh=True`

The vast CLI translates `--ssh --direct` internally to `runtype="ssh_direc ssh_proxy"`. The SDK has **no** `ssh` kwarg — `vast.create_instance(ssh=True)` fails with `TypeError: unexpected keyword argument`. Pass `runtype="ssh_direc ssh_proxy"` directly.

### `vast.execute()` only works on stopped instances

On running instances you get `400 Bad Request, Execute command only avail on stopped instances`. Use SSH for live commands.

### Offers churn fast

A specific offer ID can disappear in 1-2 minutes. The auto-pick is fine — but if you want to launch on a specific ID, re-validate with `search_offers(query=f'id={ID}')` first.

### Port allocation can be broken on a host

Symptom: `Bind for :::25723 failed: port is already allocated`. The host has a stuck Docker port allocator or a lingering container. Only fix is destroy + try another offer.

### `show_instance` returns a row, `show_instances` returns a list

The launcher handles both, but if you write polling logic: `show_instance(id=...)` returns the row dict directly, not wrapped in `{"instances": …}`.

### `show_instances()` is deprecated

Throws `DeprecationWarning`. New endpoint is `show_instances_v1(params)` with pagination. Both work as of April 2026.

---

## llama.cpp gotchas

### `--flash-attn` requires an explicit value (newer versions)

Old (pre-b8000): `--flash-attn` (boolean).
New: `--flash-attn on|off|auto`.

Without a value the parser consumes the next token. We had a container crash from this:

```
error while handling argument "--flash-attn": error: unknown value for
--flash-attn: '--jinja'
```

Always use `--flash-attn on`.

### Continuous batching is on by default

Even without `--parallel N`, requests are pipelined. Default slot count appears to be ~4 (measured behaviorally). For >4 users set `--parallel` explicitly.

### KV-cache q8 quantization halves footprint

`--cache-type-k q8_0 --cache-type-v q8_0` quantizes the KV cache from f16 to q8, halving its memory footprint with negligible quality drop on Qwen3.6. Without these flags `n_ctx` would need 2× as much VRAM.

### `--chat-template-kwargs '{"preserve_thinking":true}'`

Activates Qwen3.6's reasoning traces in `reasoning_content`. Bash quoting is fragile — the launcher uses `'{{\"preserve_thinking\":true}}'` in a Python f-string, the rendered argv token is `'{"preserve_thinking":true}'`.

### Model spec with `:tag`

`-hf unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL` — the part after `:` is the quantization variant. llama-server matches fuzzily — `Q5_K_XL` matches `Qwen3.6-27B-UD-Q5_K_XL.gguf` in the repo.

### `--metrics` enables Prometheus

Exposed at `:8080/metrics`. Useful for live tracking during workshops.

### `/v1/models` is not auth-protected

When `--api-key` is set, llama.cpp protects inference endpoints (`/v1/chat/completions`, `/completion`) but not metadata (`/v1/models`, `/health`, `/metrics`). This is upstream behavior, not a bug.

---

## Silent model download

llama-server's `-hf` loader produces **no progress logs**. After:

```
common_download_file_single_online: HEAD failed, status: 404
no remote preset found, skipping
```

(both expected — the 404 is a manifest probe, the preset lookup is optional), the container output goes silent until the model is fully loaded — 5-15 minutes on a fresh download.

This will make you think the server hung. To verify it's actually progressing:

```bash
# via SSH on the running instance
ssh -p $PORT root@$IP \
  'stat -c%s /workspace/hf_cache/hub/models--unsloth--Qwen3.6-27B-GGUF/blobs/*.downloadInProgress'
```

The file size should grow steadily. If it stops growing for >60 s, something is wrong (check `nvidia-smi`, check `dmesg`, check logs).

---

## Performance

Setup: 32 GB RTX 5090, Qwen3.6-27B Q5_K_XL, q8 KV-cache, Flash-Attn 2.

### Per-request speed

|  | tok/s |
|---|---|
| Single-user generation | **~53** |
| Single-user prompt prefill | ~161 |

### Concurrent load — default vs workshop

| Concurrency | TTFT default (parallel=4) | TTFT workshop (parallel=8) | tok/s / user | Aggregate |
|---|---|---|---|---|
| 1 | 0.7 s | 0.8 s | 53 | 44 |
| 2 | 0.8 s | 0.8 s | 42 | 73 |
| 4 | 1.0 s | 0.8 s | 25 | 90 |
| **8** | **5.1 s** | **1.0 s** | 14.5 (workshop) / 24 (default) | **108** |

Findings:

- **Aggregate throughput ceiling is ~108 tok/s** on the 5090 with this model. That's compute saturation — more slots only deepen the queue, they don't add throughput.
- **Sweet spot for workshops: `parallel=8`.** More just makes TTFT worse without aggregate gain.
- **Long conversations need `--solo-mode`** (64 k per slot). The default 16 k per slot runs out for code editing past ~6 k tokens of conversation.

---

## VRAM math

**Qwen3.6 model limit:** 256 k tokens trained (`n_ctx_train = 262144`). Above that only with YaRN scaling + quality drop.

**Per-token KV cost (q8 quantized, measured):** **~34 KB per token**, total across all slots. So 1 k tokens = 34 MB, 32 k = 1.1 GB, 128 k = 4.3 GB.

**Model footprint Qwen3.6-27B Q5_K_XL:** ~22 GB in VRAM.

**Available budget on a 32 GB card:** 32 − 22 (model) − ~1 (activations) = **~9 GB for KV cache**.

| Total ctx | KV cache | VRAM total | Reserve | Verdict |
|---|---|---|---|---|
| 64 k (default) | 2.2 GB | 24 GB | 8 GB | comfortable |
| 128 k (workshop / solo) | 4.3 GB | 26 GB | 6 GB | safe |
| 192 k | 6.5 GB | 28 GB | 4 GB | ok |
| 224 k | 7.6 GB | 29 GB | 3 GB | tight |
| 256 k (model max) | 8.7 GB | 31 GB | ~400 MB | theoretical only |

**Beyond 32 GB:** Q4_K_XL instead of Q5_K_XL saves ~6 GB (16 GB instead of 22 GB), so 320 k+ total context fits. Quality drop is small on Qwen3.6.

**Other cards (for orientation):**
- RTX 6000 Ada / 5880 Ada (48 GB): 256 k easy, +20 GB headroom
- H100 / H200 (80 GB+): 256 k easy, can load Q8_0 model (more quality)

---

## SSH debug recipes

### Bootstrap SSH access on a fresh instance

```bash
# 1. Generate keypair locally
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/vast_workspace -N ""

# 2. Register public key with vast
PUB=$(cat ~/.ssh/vast_workspace.pub)
python3 -c "
import os; from vastai import VastAI
v = VastAI(api_key=os.environ['VAST_API_KEY'])
print(v.create_ssh_key(ssh_key='$PUB'))
"

# 3. Attach key to instance (sshd reloads automatically)
python3 -c "
import os; from vastai import VastAI
v = VastAI(api_key=os.environ['VAST_API_KEY'])
pub = open(os.path.expanduser('~/.ssh/vast_workspace.pub')).read().strip()
print(v.attach_ssh(instance_id=INSTANCE_ID, ssh_key=pub))
"

# 4. Connect
# Get host port from show_instance(): inst['ports']['22/tcp'][0]['HostPort']
ssh -i ~/.ssh/vast_workspace -p HOST_PORT root@PUBLIC_IP
```

(Note: `--min-real-mbps > 0` automates steps 1-3 with a separate keypair at `~/.ssh/gpu_summon_smoketest`.)

### Useful inside the container

```bash
# What is llama-server doing?
cat /proc/$(pgrep -f /app/llama-server)/wchan
ls -la /proc/$(pgrep -f /app/llama-server)/fd/ | grep -E "blobs|gguf|socket"

# Model download progress
ls -la /workspace/hf_cache/hub/models--*/blobs/*.downloadInProgress

# GPU load
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

# Port bindings
ss -ntlp | grep 8080

# llama-server log
tail -f /var/log/llama-server.log
```

---

## Manual llama-server restart

If you want to change args without recreating the instance, kill + setsid relaunch with the **correct env**:

```bash
pkill -9 -f /app/llama-server; sleep 3
cd /app
HF_HOME=/workspace/hf_cache LD_LIBRARY_PATH=/app setsid nohup /app/llama-server \
    -hf unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL \
    --host 0.0.0.0 --port 8080 \
    --n-gpu-layers 999 --ctx-size 131072 --parallel 2 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja \
    --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --min-p 0.0 \
    --metrics \
    > /var/log/llama-server-restart.log 2>&1 < /dev/null &
disown
```

**Critical:**

- `LD_LIBRARY_PATH=/app` — otherwise `error while loading shared libraries: libllama-common.so.0`
- `HF_HOME=/workspace/hf_cache` — otherwise it re-downloads 21 GB into a fresh cache
- `setsid nohup … < /dev/null &; disown` — otherwise SSH disconnect kills the process

### SSH-disconnect bug with `pkill`

`pkill -f` sometimes matches ancestor processes of your own SSH session and disconnects you. Workaround: kill by specific PID (`pgrep -f` first, then `kill -9 PID`), not via `pkill`.

---

## Cost reference

- Vast.ai bills **per second** from `actual_status="running"`
- Default filters cap compute at $0.50/h and bandwidth at $0.01/GB
- Model pull (~21 GB Q5_K_XL) is a one-time transfer per launch
- Disk storage on top, ~$0.20/GB/month pro-rated (≈ $0.011/h for 80 GB)
- Realistic total **$0.32–0.40/h** for 32 GB RTX 5090 incl. storage

Typical sessions:

| Session | Compute | Transfer | Storage | Total |
|---|---|---|---|---|
| 30 min test (cold start) | $0.15 | $0.06 (model pull) | $0.01 | **$0.22** |
| 2 h work session | $0.60 | $0.06 | $0.02 | **$0.68** |
| 4 h workshop | $1.20 | $0.06 | $0.05 | **$1.31** |
| 24 h continuous | $7.20 | $0.06 | $0.30 | **$7.56** |

**Cost hygiene:**

- **Every run ends with `python summon.py --destroy ID`** — vast has no auto-stop on idle
- `python summon.py --list` shows what's still running
- For broken hosts (port conflicts etc.): destroy and try a new offer. Lost setup time is cheaper than waiting on a dead host.

---

## vast.ai Templates

A vast.ai Template captures `image + onstart_cmd + runtype` so you don't have to re-render them on every launch. The launcher supports them via `--template-hash`.

### Creating a template from an existing instance

```python
from vastai import VastAI
import os
v = VastAI(api_key=os.environ['VAST_API_KEY'])

# Onstart with mode-aware env (LLAMA_PARALLEL, LLAMA_CTX, LLAMA_MODEL,
# LLAMA_API_KEY are injected per launch; defaults shown below).
ONSTART = '''/app/llama-server \\
  -hf ${LLAMA_MODEL:-unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL} \\
  --host 0.0.0.0 --port 8080 \\
  --n-gpu-layers 999 \\
  --ctx-size ${LLAMA_CTX:-65536} \\
  --parallel ${LLAMA_PARALLEL:-4} \\
  --cache-type-k q8_0 --cache-type-v q8_0 \\
  --flash-attn on --jinja \\
  --chat-template-kwargs '{"preserve_thinking":true}' \\
  --temp 0.7 --top-p 0.8 --top-k 20 \\
  --presence-penalty 1.5 --min-p 0.0 \\
  ${LLAMA_API_KEY:+--api-key $LLAMA_API_KEY} \\
  --metrics \\
  2>&1 | tee /var/log/llama-server.log'''

result = v.create_template(
    name='qwen3.6-27b-q5kxl',
    image='ghcr.io/ggml-org/llama.cpp:server-cuda',
    onstart=ONSTART,
    ssh=True, direct=True,
    env={'-p 8080:8080': '1', 'HF_HOME': '/workspace/hf_cache'},
)
print(result['template']['hash_id'])
```

Note: `create_template` accepts `ssh=True, direct=True` (these are translated). `update_template` does **not** translate — pass `use_ssh=True, ssh_direct=True, runtype="ssh"` directly to it.

### Using a template

```bash
# Pin in env
GPU_SUMMON_TEMPLATE_HASH=<hash> python summon.py --solo-mode

# Or per-launch
python summon.py --template-hash <hash> --solo-mode
```

The launcher will use the template's image + onstart, and inject `LLAMA_PARALLEL`, `LLAMA_CTX`, `LLAMA_MODEL`, `LLAMA_API_KEY` env so the same template handles all modes.

### When templates help

- Multiple launches with stable image / onstart, only mode varies
- Sharing a known-good config with others (template hashes are public-shareable)
- Avoiding `onstart_cmd` quoting issues during creation

### Caveats

- Template env doesn't merge with create-time env — you must pass `-p 8080:8080` and `HF_HOME` again on `create_instance`
- Updating a template across SDK / API has different keyword names (see [vast.ai API quirks](#vastai-api-quirks))

---

## code-server Mode (--with-codeserver)

The `--with-codeserver` flow runs llama-server *and* code-server (browser-VS-Code) on the same rented box, fronted by Caddy with a duckdns wildcard cert. The original design tried to run xaresaicoder via Docker-in-Docker — that turned out impossible on vast.ai (see DinD subsection below), so we pivoted to a single shared code-server (no per-project workspace isolation, but actually works).

### vast.ai forbids `--privileged`, so DinD is dead on arrival

**This is the biggest gotcha** — and we paid for the lesson. vast.ai's SDK and HTTP API expose no way to launch an instance with `--privileged` or `--cap-add NET_ADMIN`. The CLI's `instance create` command has no flag for it. By design, vast hosts run untrusted user containers without those capabilities for security on shared hosts.

Symptom of trying anyway: `dockerd` starts, fails on the bridge-network setup, exits within 1 second:

```
failed to start daemon: Error initializing network controller: error obtaining
controller instance: failed to register "bridge" driver: failed to create NAT
chain DOCKER: iptables failed: ... Could not fetch rule set generation id:
Permission denied (you must be root)
```

Workarounds we considered and rejected:
- **Rootless dockerd**: works without privileged but is slow (vfs storage, no overlay), complex to set up, and xaresaicoder wasn't built for it.
- **`dockerd --iptables=false`**: dockerd starts but containers can't talk to each other or the outside without manual `--network=host` everywhere → port collisions everywhere.
- **`sysbox` runtime**: requires the host to install it, vast hosts won't.
- **Different cloud (RunPod, Lambda)**: works but out of scope for a vast.ai-only launcher.

The pivot: drop xaresaicoder's container-per-workspace model, install code-server natively in the gpu-summon image, run it as a host process alongside llama-server. Lose the project-isolation feature, gain "actually works on vast.ai." Caddy + duckdns wildcard TLS still on top, code-server's built-in `PASSWORD` env handles login.

If you ever need true per-workspace isolation on vast.ai, your options are: (a) different provider, (b) accept that vast is for unprivileged single-user workloads only.

### vast.ai's `/.launch` apt-installs ssh/tmux/rsync at boot

Every vast.ai container is wrapped by a vast-injected `/.launch` script that runs **before** your `onstart_cmd`. It does `apt-get update && apt-get install -y openssh-server tmux rsync software-properties-common` so vast's web-console SSH-button works.

If your image is slim (e.g. `nvidia/cuda:*-runtime-ubuntu22.04`), those packages are not pre-installed. If the rented host can't reach `archive.ubuntu.com` at boot (we observed `Connection failed [IP: 91.189.91.81 80]` on a real launch), apt fails with "held broken packages" — and `/.launch` then enters an infinite loop:

```
/.launch: line 38: /usr/sbin/sshd: No such file or directory
/.launch: line 48: ssh: command not found
[repeats forever]
```

Your `onstart_cmd` never runs. The container looks "running" to the vast API but is dead.

**Fix:** pre-install the packages in your Dockerfile so `/.launch`'s apt-installs become idempotent no-ops. Specifically: `openssh-server openssh-client openssh-sftp-server rsync tmux software-properties-common`. Cost: ~150 MB of image size. Worth it.

### llama-server binaries need their build-time GLIBC

The `ghcr.io/ggml-org/llama.cpp:server-cuda` image ships `llama-server` compiled against the GLIBC of its own base distro (currently Ubuntu 24.04 = GLIBC 2.39, also `GLIBCXX_3.4.32`). Copying just `/app` over to a slimmer `nvidia/cuda:*-runtime-ubuntu22.04` (GLIBC 2.35) image makes the binary fail to load with:

```
/app/llama-server: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
/app/llama-server: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

The container appears healthy (other processes start), llama-server silently never comes up, and any readiness probe against its port hangs forever. Fix: use `ghcr.io/ggml-org/llama.cpp:server-cuda` as the **base** (FROM), not as a stage to copy from. The CUDA + glibc combination is then guaranteed compatible.

### duckdns supports wildcards (verified empirically)

Any `*.<your-name>.duckdns.org` resolves to the same A record as `<your-name>.duckdns.org`. Not documented in their FAQ but works in practice — `nslookup llm.xaicoder.duckdns.org` returned the same A as the apex during a real launch. One duckdns subdomain is enough for both `${DOMAIN}` and `llm.${DOMAIN}`.

### Let's Encrypt wildcard cert needs DNS-01

HTTP-01 doesn't work for wildcards. We use Caddy's `caddy-dns/duckdns` plugin (compiled into the binary in `codeserver/Dockerfile`). Caddy reads `DUCKDNS_TOKEN` from env and handles the TXT record dance automatically. Renewal is also automatic — no cron needed.

The first cert acquisition takes 30-60 s; readiness probes against `https://${DOMAIN}/` will get connection-refused or SSL-handshake errors during that window. `wait_until_ready` catches `SSLError` alongside the connection errors so this isn't fatal.

duckdns supports only one TXT record value per subdomain at a time. Caddy issues separate certs for the apex block and the wildcard block, so the two ACME DNS-01 challenges run sequentially — the plugin clears the TXT between them. We've seen one transient "TXT did not propagate" retry in the wild but it self-recovers within ~30 s.

### Disable Caddy's TXT-propagation self-check on vast.ai

Caddy's default DNS-challenge flow does a self-check before asking the ACME server to validate: it queries duckdns' authoritative nameservers directly (TCP/53 to e.g. `35.183.157.249`). vast.ai hosts often firewall outbound TCP/53 to specific upstream IPs — the result is `dial tcp 35.183.157.249:53: i/o timeout`, Caddy declares the challenge "not ready," falls through LE-prod → LE-staging → ZeroSSL until one happens to succeed before its timeout. First-launch SSL errors and ~5-10 minute apex-cert acquisition were the symptom.

Fix in `codeserver/Caddyfile.template`: define an `(acme_dns_config)` snippet imported by both site blocks, with `propagation_delay 60s` + `propagation_timeout -1`. The `-1` disables the self-check; the `60s` delay is just enough for duckdns + LE/ZeroSSL to see the same TXT. Cert acquisition becomes deterministic at ~60-90 s instead of "minutes, maybe."

### Public Suffix List == own rate limit bucket

`duckdns.org` is on the [Public Suffix List](https://publicsuffix.org/list/), so each `*.duckdns.org` subdomain counts as its own "registered domain" for Let's Encrypt rate limiting (50 certs/week, 5 duplicate certs/week). You won't share quota with the rest of the duckdns user base.

### Port 443 is essentially never free on vast.ai hosts

Vast.ai instances are containers; mapping `-p 443:443` only works if the underlying host has port 443 free. **In practice, vast hosts essentially never do** — the host runs its own web service on 443 (vast-internal management or SSH proxy). We tried 4+ offers in a row across different geographies; none gave host_port=443 for container 443. Same for 80 and 8080.

So the container's port 443 always gets mapped to a random host port like 44140. `https://${DOMAIN}/` (default port 443) lands on vast's own service, not your Caddy. The launcher accepts this and bakes `:PORT` into all user-facing URLs (FERTIG output, opencode config, readiness probes). Bookmark with the port; one minor uglyness for an otherwise working setup.

Earlier the launcher tried `PORT_443_RETRY_LIMIT=3` offers hoping one would have :443 free. We removed that retry — burns ~$0.05 per attempt × ~3min × 3 attempts for ~0% chance of success.

### Disk math for `--with-codeserver`

| Component | Size |
|---|---|
| Custom image (CUDA runtime + caddy + code-server + llama.cpp) | ~5 GB |
| Q5_K_XL model cache | ~22 GB |
| code-server extensions + user data | 1-3 GB |
| Project files | bring-your-own |
| **Floor** | **~60 GB disk** |

The launcher auto-bumps `--disk` to 60 GB when `--with-codeserver` is set.

### RAM math (system, not VRAM)

22 GB model in VRAM doesn't touch system RAM, but the IDE process does:

| Component | Idle |
|---|---|
| Caddy | ~50 MB |
| code-server + extensions | ~500-800 MB (more under active use) |
| **Floor** | **~1 GB system RAM, plus a few GB per running language server / build job** |

Vast machines with 32-GB GPUs typically come with 32-64 GB system RAM and 8-16 vCPUs, so headroom is comfortable for normal dev work.
