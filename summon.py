#!/usr/bin/env python3
"""
gpu-summon (summon.py) - Rent a cloud GPU, serve a local LLM
------------------------------------------------------------
Default: llama.cpp/llama-server with Unsloth GGUF quants (for Qwen3.6+)
Optional: Ollama (for older models that don't need GGUF)

Currently uses vast.ai as the GPU marketplace.

Requirements:
    pip install vastai requests

Environment:
    VAST_API_KEY   - https://cloud.vast.ai/manage-keys/

Examples:
    # Qwen3.6-27B with Unsloth Q5_K_XL on a 32GB GPU
    python summon.py --model unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL --min-vram 32

    # Q8_0 (bigger, best quality) - needs 32GB+
    python summon.py --model unsloth/Qwen3.6-27B-GGUF:Q8_0 --min-vram 32 --max-price 0.60

    # Q4 for smaller budget on a 24GB card
    python summon.py --model unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL --min-vram 24 --max-price 0.35

    # Ollama fallback for older models
    python summon.py --backend ollama --model qwen3-coder:30b --min-vram 24

    # Cleanup
    python summon.py --list
    python summon.py --destroy 12345678
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import requests

try:
    from vastai import VastAI
except ImportError:
    print("ERROR: vastai SDK not installed. Run: pip install vastai")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Backend definitions
# -----------------------------------------------------------------------------

# As of April 2026, CUDA 13.2 has a known bug that produces gibberish output
# with many models (especially Qwen3.6). We actively filter it out.
CUDA_BLACKLIST = ["13.2"]
CUDA_MIN_GOOD = 12.0      # lower works, but FP8/Flash-Attn wants 12.0+

BACKENDS = {
    "llamacpp": {
        # Official llama.cpp CUDA server image
        "image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
        "exposed_port": 8080,
        "api_path": "/v1",
        "default_model": "unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL",
        # Important: --jinja for tool calling (opencode), --chat-template-kwargs
        # for Qwen3.6 Preserved Thinking, plus Unsloth's sampling defaults.
        # ctx-size is the TOTAL budget, not per-slot. With parallel=N each
        # slot gets ctx-size/N. Default 65536 + parallel=4 -> 16k per user,
        # which is enough for normal chat conversations.
        # --api-key secures the endpoint via Bearer token. Without a key,
        # llama-server is reachable PUBLICLY - anyone with IP:PORT can run
        # inference on your bill. {API_KEY} is injected by the script.
        "onstart_template": (
            "/app/llama-server "
            "-hf {MODEL} "
            "--host 0.0.0.0 --port 8080 "
            "--n-gpu-layers 999 "
            "--ctx-size {CTX} "
            "--parallel {PARALLEL} "
            "--cache-type-k q8_0 --cache-type-v q8_0 "
            "--flash-attn on "
            "--jinja "
            "--chat-template-kwargs '{{\"preserve_thinking\":true}}' "
            "--temp 0.7 --top-p 0.8 --top-k 20 "
            "--presence-penalty 1.5 --min-p 0.0 "
            "--api-key {API_KEY} "
            "--metrics "
            "2>&1 | tee /var/log/llama-server.log"
        ),
        "ready_check": "http://{host}:{port}/v1/models",
        "disk_gb_min": 80,
    },
    "ollama": {
        "image": "ghcr.io/open-webui/open-webui:ollama",
        "exposed_port": 11434,
        "api_path": "/v1",
        "default_model": "qwen3-coder:30b",
        "onstart_template": (
            "while ! curl -s http://localhost:11434/api/tags > /dev/null; do "
            "  sleep 3; "
            "done && echo 'Ollama ready'"
        ),
        "ready_check": "http://{host}:{port}/api/tags",
        "disk_gb_min": 60,
    },
    # code-server: llama-server + browser-based VS Code (code-server) +
    # Caddy with duckdns wildcard TLS. Everything runs as a host process -
    # NO Docker-in-Docker, because vast.ai's standard offers don't allow
    # --privileged (see LESSONS.md "vast.ai forbids privileged mode").
    # Not selectable via --backend - activated through --with-codeserver.
    "codeserver": {
        "image": "ghcr.io/dg1001/gpu-summon-codeserver:latest",
        "exposed_port": 443,        # Caddy/HTTPS - the user-facing endpoint
        "extra_ports": [8080, 80],  # 8080: direct LLM (legacy + opencode)
                                    # 80:   ACME HTTP fallback, unused but harmless
        "api_path": "/v1",
        "default_model": "unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL",
        # /opt/onstart.sh is set as CMD in the image, but vast expects an
        # explicit onstart_cmd. Output to a file so it's debuggable via SSH.
        "onstart_template": "/opt/onstart.sh > /var/log/onstart.log 2>&1",
        # code-server's login page returns 200 (HTML) without auth, so the
        # apex itself works as a readiness check. Validates in one shot:
        # duckdns DNS + Caddy cert + Caddy routing + code-server up.
        "ready_check": "https://{domain}/",
        "disk_gb_min": 60,          # llama cache + code-server + margin
    },
}


# -----------------------------------------------------------------------------
# Offer search
# -----------------------------------------------------------------------------

def _run_search(vast: VastAI, query: str, verbose: bool = True,
                cuda_min: float = 12.0) -> list:
    """Helper: runs the query and returns a filtered offers list.

    CUDA filtering happens Python-side, NOT in the query. Reason:
    vast.ai compares cuda_max_good in queries partially lexicographically
    (as strings), so '12.10' is falsely smaller than '12.4' and legitimate
    hosts get dropped. Floats compare correctly Python-side.
    """
    if verbose:
        print(f"[search] Query: {query}")
    try:
        result = vast.search_offers(
            query=query,
            order="dlperf_per_dphtotal-",
            limit="30",
        )
    except Exception as e:
        print(f"[search] ERROR in search_offers: {e}")
        return []
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print(f"[search] Could not parse response: {result[:200]}")
            return []
    raw = result.get("offers", []) if isinstance(result, dict) else result

    if verbose and raw:
        print(f"[search]   -> {len(raw)} raw hits from the API")

    # CUDA filtering Python-side: min version + 13.2 blacklist
    def cuda_ok(o):
        try:
            cv = float(o.get("cuda_max_good", 0))
        except (ValueError, TypeError):
            return False
        return cv >= cuda_min and str(o.get("cuda_max_good")) not in CUDA_BLACKLIST

    filtered = [o for o in raw if cuda_ok(o)]
    if verbose and len(raw) != len(filtered):
        print(f"[search]   -> {len(filtered)} after CUDA filter "
              f"(min {cuda_min}, blacklist {CUDA_BLACKLIST})")
    return filtered


def find_best_offers(vast: VastAI, *,
                    min_vram_gb: int = 32,
                    max_dph: float = 0.50,
                    min_reliability: float = 0.95,
                    min_inet_down: int = 200,
                    max_inet_cost: float = 0.01,
                    region: str | None = None,
                    cuda_min: float = 12.0,
                    verbose: bool = True) -> list[dict]:
    """
    Find a matching GPU with a "cheap but stable" filter.

    Pitfalls handled here:
    - 'reliability' in the query, 'reliability2' in the response (vast.ai inconsistency)
    - gpu_ram in the QUERY is GB (float allowed), but MB in the RESPONSE.
      Writing 30000 here would search for cards with 30 TB VRAM -> 0 hits.
    - gpu_ram nominal vs reserved: 32GB cards often report 32510 instead of 32768
    - cuda_max_good in queries compares partially lexicographically -> filter Python-side
    - CUDA 13.2 produces gibberish on Qwen3.6 -> blacklist
    """
    # Query: GB (float), with 7% buffer for cards that report slightly less
    gpu_ram_query_threshold = round(min_vram_gb * 0.93, 2)
    # Post-filter: MB, stricter (5% buffer) against the response
    gpu_ram_real_threshold = int(min_vram_gb * 1000 * 0.95)

    # inet_down_cost belongs in base: it's a cost cap like dph_total.
    # Some hosts charge $0.04/GB - already $1 on a 25 GB model pull.
    base_filters = [
        f"gpu_ram>={gpu_ram_query_threshold}",
        f"dph_total<={max_dph}",
        f"inet_down_cost<={max_inet_cost}",
        "rentable=true",
        "rented=false",
        "num_gpus=1",
    ]
    quality_filters = [
        f"reliability>={min_reliability}",
        "verified=true",
        f"inet_down>={min_inet_down}",
        "direct_port_count>=2",
    ]
    if region:
        base_filters.append(f'geolocation={region}')

    def post_filter(offers: list) -> list:
        """Python-side: drop cards with too little GPU RAM."""
        return [o for o in offers
                if o.get("gpu_ram", 0) >= gpu_ram_real_threshold]

    def search(filters):
        query = " ".join(filters)
        return post_filter(_run_search(vast, query, verbose, cuda_min=cuda_min))

    # Attempt 1: all filters
    offers = search(base_filters + quality_filters)
    if verbose and offers:
        print(f"[search] Attempt 1: {len(offers)} hits after all filters")

    # Attempt 2: without bandwidth/port filters
    if not offers:
        if verbose:
            print("[search] No hits - relaxing inet_down/direct_port...")
        relaxed = [f for f in quality_filters
                   if not f.startswith(("inet_down", "direct_port_count"))]
        offers = search(base_filters + relaxed)

    # Attempt 3: only base + verified
    if not offers:
        if verbose:
            print("[search] Still nothing - only verified+rentable...")
        offers = search(base_filters + ["verified=true"])

    # Attempt 4: drop all filters, only base
    if not offers:
        if verbose:
            print("[search] Last attempt - only base filters...")
        offers = search(base_filters)

    if not offers:
        print("[search] No matching offers found.")
        print("         Tip: raise --max-price, lower --min-vram, "
              "or drop --region.")
        return []

    if verbose:
        print(f"[search] {len(offers)} offers found, top 5:")
        for o in offers[:5]:
            inet_c = o.get('inet_down_cost') or 0
            print(f"   id={o['id']:>10}  {o['gpu_name']:<15} "
                  f"VRAM={o['gpu_ram']/1024:>5.1f}GB  "
                  f"${o['dph_total']:>5.3f}/h  "
                  f"rel={o['reliability2']:.3f}  "
                  f"cuda={o.get('cuda_max_good', '?')}  "
                  f"inet={o.get('inet_down', 0):>5.0f}Mbps "
                  f"@${inet_c:.4f}/GB  "
                  f"geo={o.get('geolocation', '?')}")

    # Top-5 sorted by reliability: first element is the primary pick,
    # the rest are backups in case the smoketest fails.
    top = sorted(offers[:5], key=lambda o: (-o['reliability2'], o['dph_total']))
    if verbose:
        print("[search] Order by reliability:")
        for i, o in enumerate(top, 1):
            print(f"   {i}. id={o['id']} {o['gpu_name']}  "
                  f"${o['dph_total']:.3f}/h  rel={o['reliability2']:.3f}  "
                  f"inet_adv={o.get('inet_down', 0):.0f}Mbps")
    return top


def find_best_offer(vast: VastAI, **kwargs) -> dict | None:
    """Backward-compat: first element of the find_best_offers list."""
    offers = find_best_offers(vast, **kwargs)
    return offers[0] if offers else None


# -----------------------------------------------------------------------------
# Launch instance
# -----------------------------------------------------------------------------

def create_instance(vast: VastAI, offer: dict, backend: str, model: str, *,
                    disk_gb: int = 80, ctx: int = 65536, parallel: int = 4,
                    label: str = "opencode-llm",
                    api_key: str = "",
                    template_hash: str | None = None,
                    extra_env: dict | None = None,
                    image_override: str | None = None) -> int:
    """Create the instance with the desired backend image.

    api_key: Bearer token that llama-server accepts. Empty = no auth
    (endpoint public). Auto-generated by cmd_launch when unset.
    template_hash: optional, hash of a vast.ai template. If set,
    image + onstart + runtype come from the template, and the script
    only injects mode-specific env (LLAMA_PARALLEL/LLAMA_CTX/LLAMA_MODEL/
    LLAMA_API_KEY). Currently only supported for the llamacpp backend
    with env-aware onstart.
    extra_env: when backend='codeserver', the pre-built env dict from
    build_codeserver_env() (DOMAIN, DUCKDNS_TOKEN, CODESERVER_PASSWORD,
    port mappings). Built by the caller, just passed through here.
    image_override: overrides cfg['image'] (for --code-image custom
    builds). Only relevant when backend='codeserver'.
    """
    cfg = BACKENDS[backend]
    use_template = template_hash is not None

    if use_template and backend != "llamacpp":
        raise ValueError(
            f"--template-hash is currently only supported with backend=llamacpp, "
            f"not {backend}")

    if backend == "codeserver" and extra_env is None:
        raise ValueError("backend=codeserver requires the extra_env parameter")

    image = image_override or cfg["image"]

    print(f"[create] Backend: {backend}")
    print(f"[create] Model: {model}")
    if backend == "llamacpp":
        print(f"[create] ctx={ctx} parallel={parallel} "
              f"({ctx//parallel} tokens per slot)")
    if use_template:
        print(f"[create] Template: {template_hash} "
              f"(image+onstart from template)")
    else:
        print(f"[create] Image: {image}")
    print(f"[create] Creating instance on offer {offer['id']}...")

    if backend == "codeserver":
        env = dict(extra_env)  # copy so we don't mutate caller's dict
    else:
        env = {
            f"-p {cfg['exposed_port']}:{cfg['exposed_port']}": "1",
        }
    if backend == "llamacpp":
        env["HF_HOME"] = "/workspace/hf_cache"
        if use_template:
            # Template's onstart reads these envs with defaults
            env["LLAMA_PARALLEL"] = str(parallel)
            env["LLAMA_CTX"] = str(ctx)
            env["LLAMA_MODEL"] = model
            if api_key:
                env["LLAMA_API_KEY"] = api_key
    elif backend == "ollama":
        env["-p 8080:8080"] = "1"
        env["OLLAMA_HOST"] = "0.0.0.0:11434"
        env["OLLAMA_KEEP_ALIVE"] = "30m"
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    # backend == "codeserver": env is already complete from build_codeserver_env

    if use_template:
        # Template provides image + onstart_cmd + runtype.
        # Env is set completely here (vast does NOT merge with template env,
        # it replaces). That's why -p, HF_HOME etc. are included.
        result = vast.create_instance(
            id=offer["id"],
            disk=disk_gb,
            env=env,
            template_hash=template_hash,
            label=label,
        )
    else:
        # Self-contained path: render image + onstart_cmd ourselves
        if backend == "llamacpp":
            onstart = cfg["onstart_template"].format(
                MODEL=model, CTX=ctx, PARALLEL=parallel, API_KEY=api_key or "")
        elif backend == "codeserver":
            # /opt/onstart.sh is set as CMD in the image; vast still expects
            # an explicit onstart_cmd. Template has no placeholders - all
            # values come via env.
            onstart = cfg["onstart_template"]
        else:
            onstart = cfg["onstart_template"].format(MODEL=model, CTX=ctx)
        # The SDK does NOT accept ssh/direct as kwargs - the CLI translates
        # --ssh --direct internally to runtype="ssh_direc ssh_proxy".
        # Writing ssh=True here would raise a TypeError.
        result = vast.create_instance(
            id=offer["id"],
            image=image,
            disk=disk_gb,
            env=env,
            onstart_cmd=onstart,
            runtype="ssh_direc ssh_proxy",
            label=label,
        )
    if isinstance(result, str):
        result = json.loads(result)
    # Vast quirk: sometimes returns {success: False, new_contract: N,
    # instance_api_key: ...} even though the instance was actually
    # created. If new_contract is present we treat that as success
    # (otherwise we get orphan instances that keep running on the bill).
    if not result.get("success"):
        if result.get("new_contract"):
            print(f"[create] vast quirk: success=False with new_contract="
                  f"{result['new_contract']} - accepting as created.")
        else:
            raise RuntimeError(f"create_instance failed: {result}")

    instance_id = result["new_contract"]
    print(f"[create] Instance ID: {instance_id}")
    return instance_id


def wait_until_running(vast: VastAI, instance_id: int,
                       timeout_sec: int = 600) -> dict:
    """Poll until the instance is 'running' and ports are mapped."""
    print(f"[wait]  Waiting for 'running' status (max {timeout_sec}s)...")
    start = time.time()
    last_status = None

    while time.time() - start < timeout_sec:
        try:
            inst_raw = vast.show_instance(id=instance_id)
            if isinstance(inst_raw, str):
                inst_raw = json.loads(inst_raw)
            inst = inst_raw.get("instances", inst_raw)
            if isinstance(inst, list):
                inst = inst[0] if inst else {}

            status = inst.get("actual_status") or inst.get("cur_state")
            if status != last_status:
                elapsed = int(time.time() - start)
                print(f"[wait]  [{elapsed:>3}s] Status: {status}")
                last_status = status

            if status in ("exited", "offline", "unknown"):
                raise RuntimeError(
                    f"Instance in error state: {status}. "
                    f"Check logs or try a new offer."
                )

            if status == "running" and inst.get("ports"):
                print(f"[wait]  Ports active: {list(inst['ports'].keys())}")
                return inst

            time.sleep(5)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[wait]  WARN: {e}")
            time.sleep(5)

    raise TimeoutError(f"Instance {instance_id} not ready within {timeout_sec}s")


def get_endpoint(inst: dict, backend: str) -> tuple[str, int]:
    """Read IP and mapped port for the selected backend."""
    ip = inst.get("public_ipaddr") or inst.get("ssh_host")
    ports = inst.get("ports") or {}
    internal = f"{BACKENDS[backend]['exposed_port']}/tcp"
    mapping = ports.get(internal)
    port = int(mapping[0]["HostPort"]) if mapping else None
    return ip, port


def wait_until_ready(url: str, *, headers: dict | None = None,
                     timeout_sec: int = 1800, label: str = "") -> None:
    """
    Poll an HTTP(S) URL until it answers 200.
    On first start, llama-server downloads the model from HuggingFace -
    5-15 minutes depending on model size. With --with-codeserver,
    Caddy ACME cert + code-server startup add ~30s on top.

    label: optional tag for output (e.g. 'llm' / 'code') when multiple
    probes run back-to-back.
    """
    tag = f" [{label}]" if label else ""
    print(f"[ready]{tag} Probing {url}...")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = requests.get(url, timeout=10, headers=headers, verify=True)
            if r.status_code == 200:
                print(f"[ready]{tag} API responding "
                      f"(after {int(time.time()-start)}s)")
                return
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.SSLError):
            # SSL error is expected while Caddy has no cert yet
            pass
        elapsed = int(time.time() - start)
        if elapsed % 30 < 5:
            print(f"[ready]{tag} [{elapsed}s] not ready yet, still waiting...")
        time.sleep(5)
    raise TimeoutError(f"{tag.strip() or 'API'} not ready on {url} "
                       f"after {timeout_sec}s")


# -----------------------------------------------------------------------------
# Bandwidth smoketest (measure real HF->host throughput)
# -----------------------------------------------------------------------------

def get_ssh_port(inst: dict) -> int | None:
    ports = inst.get("ports") or {}
    mapping = ports.get("22/tcp")
    return int(mapping[0]["HostPort"]) if mapping else None


def ensure_smoketest_keypair() -> tuple[str, str]:
    """Generate an ed25519 keypair for smoketest SSH if none exists yet.

    Dedicated key (~/.ssh/gpu_summon_smoketest) so it doesn't collide with
    user keys. Returns (public_key_string, private_key_path).
    """
    key_dir = Path.home() / ".ssh"
    key_dir.mkdir(parents=True, exist_ok=True)
    priv = key_dir / "gpu_summon_smoketest"
    pub = key_dir / "gpu_summon_smoketest.pub"
    if not priv.exists():
        print(f"[probe] Generating SSH key for smoketest: {priv}")
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "",
             "-C", "gpu-summon-smoketest", "-f", str(priv)],
            check=True, capture_output=True,
        )
    return pub.read_text().strip(), str(priv)


def smoketest_hf_bandwidth(vast: VastAI, instance_id: int,
                            ip: str, ssh_port: int,
                            ssh_key_pub: str, ssh_key_priv: str,
                            probe_seconds: int = 15,
                            wait_for_download_sec: int = 90) -> float:
    """Measure real HF->host bandwidth during the model download.

    Strategy: SSH in, poll until the '.downloadInProgress' cache file exists,
    sample file-size growth over probe_seconds. This measures exactly what
    matters in the end - not theoretical, but the effective HF throughput
    for this host right now.

    Returns: measured Mbps (float). 0.0 = probe failed (SSH, timeout, parse -
    all treated as 'unusable').
    """
    try:
        r = vast.attach_ssh(instance_id=instance_id, ssh_key=ssh_key_pub)
        if isinstance(r, str):
            r = json.loads(r)
        # already-associated is OK, anything else might not be.
        if not r.get("success") and \
           "already" not in str(r.get("msg", "")).lower():
            print(f"[probe] WARN attach_ssh: {r}")
    except Exception as e:
        print(f"[probe] attach_ssh error: {e}")
        return 0.0

    ssh_args = [
        "ssh",
        "-i", ssh_key_priv,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=4",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_port),
        f"root@{ip}",
    ]

    print(f"[probe] Waiting for SSH to {ip}:{ssh_port}...")
    deadline = time.time() + 60
    while time.time() < deadline:
        r = subprocess.run(ssh_args + ["echo OK"],
                           capture_output=True, timeout=10, text=True)
        if r.returncode == 0:
            break
        time.sleep(3)
    else:
        print("[probe] SSH unreachable after 60s")
        return 0.0

    print(f"[probe] Waiting for HF download to start "
          f"(max {wait_for_download_sec}s)...")
    find_cmd = ("ls -1 /workspace/hf_cache/hub/models--*/blobs/"
                "*.downloadInProgress 2>/dev/null | head -1")
    deadline = time.time() + wait_for_download_sec
    blob = ""
    while time.time() < deadline:
        r = subprocess.run(ssh_args + [find_cmd],
                           capture_output=True, text=True, timeout=10)
        out = r.stdout.strip()
        if out:
            blob = out
            break
        time.sleep(3)
    if not blob:
        print(f"[probe] No .downloadInProgress after {wait_for_download_sec}s "
              f"- download didn't start, is llama-server stuck?")
        return 0.0
    print(f"[probe] Download running: {blob}")

    print(f"[probe] Sampling {probe_seconds}s of file-size growth...")
    measure_cmd = (
        f'S1=$(stat -c%s "{blob}" 2>/dev/null || echo 0); '
        f'sleep {probe_seconds}; '
        f'S2=$(stat -c%s "{blob}" 2>/dev/null || echo 0); '
        f'echo "$S1 $S2"'
    )
    r = subprocess.run(ssh_args + [measure_cmd],
                       capture_output=True, text=True,
                       timeout=probe_seconds + 30)
    if r.returncode != 0:
        print(f"[probe] Measurement SSH command failed: {r.stderr.strip()}")
        return 0.0
    try:
        s1_str, s2_str = r.stdout.strip().split()
        s1, s2 = int(s1_str), int(s2_str)
        bps = max(0, s2 - s1) / probe_seconds
        return bps * 8 / 1_000_000
    except Exception as e:
        print(f"[probe] Parse error: {e}, output={r.stdout!r}")
        return 0.0


# -----------------------------------------------------------------------------
# opencode Config
# -----------------------------------------------------------------------------

def write_opencode_config(host: str, port: int, model: str, backend: str,
                           api_key: str = "",
                           config_dir: Path | None = None,
                           code_domain: str | None = None,
                           code_port: int | None = None) -> Path:
    """Write opencode.json + auth.json for the remote setup.

    api_key: the Bearer token llama-server expects. Empty = no auth
    (a default dummy key is written; without --api-key llama-server
    accepts everything anyway).
    code_domain: with --with-codeserver (e.g. 'mybox.duckdns.org'), the
    baseURL is set to https://llm.{domain}/v1 instead of the bare IP -
    uses Caddy's TLS and hides the changing vast IP.
    code_port: actual host port for container-443. 443 = omitted (clean
    URL); random port = embedded as ":PORT".
    """
    if config_dir is None:
        config_dir = Path.home() / ".config" / "opencode"
    auth_dir = Path.home() / ".local" / "share" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    auth_dir.mkdir(parents=True, exist_ok=True)

    api_path = BACKENDS[backend]["api_path"]
    if code_domain:
        port_suffix = "" if code_port in (None, 443) else f":{code_port}"
        base_url = f"https://llm.{code_domain}{port_suffix}{api_path}"
        display_host = code_domain
    else:
        base_url = f"http://{host}:{port}{api_path}"
        display_host = host
    provider_id = f"vast-{backend}"

    # Fetch the real model name from the server so the key in opencode is right.
    # If auth is active, try with the api-key.
    model_id = model
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(f"{base_url}/models", timeout=10, headers=headers)
        if r.ok:
            models_data = r.json().get("data", [])
            if models_data:
                model_id = models_data[0]["id"]
                print(f"[cfg]   Model ID from server: {model_id}")
    except Exception as e:
        print(f"[cfg]   WARN: could not fetch model ID, using {model}: {e}")

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"Vast.ai {backend} ({display_host})",
                "options": {"baseURL": base_url},
                "models": {model_id: {"tools": True}}
            }
        },
        "model": f"{provider_id}/{model_id}",
        "small_model": f"{provider_id}/{model_id}",
    }

    config_path = config_dir / "opencode.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
            existing.setdefault("provider", {})[provider_id] = \
                config["provider"][provider_id]
            existing["model"] = config["model"]
            config = existing
        except json.JSONDecodeError:
            print("[cfg]   WARN: existing config not parsable, overwriting")

    config_path.write_text(json.dumps(config, indent=2))
    print(f"[cfg]   {config_path}")

    auth_path = auth_dir / "auth.json"
    auth = {}
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text())
        except json.JSONDecodeError:
            pass
    # If api_key is set: opencode sends it as Bearer; llama-server
    # validates. Empty = the previously used dummy value is fine because
    # llama-server without --api-key accepts everything.
    auth[provider_id] = {"type": "api", "key": api_key or backend}
    auth_path.write_text(json.dumps(auth, indent=2))
    print(f"[cfg]   {auth_path}")

    return config_path


# -----------------------------------------------------------------------------
# code-server Helpers (--with-codeserver)
# -----------------------------------------------------------------------------

DUCKDNS_UPDATE_URL = "https://www.duckdns.org/update"


def update_duckdns(subdomain: str, token: str, ip: str | None = None) -> bool:
    """Set the A-record subdomain.duckdns.org. ip=None resets to the caller's
    current connection IP (as seen by duckdns); for cleanup purposes we
    explicitly pass '0.0.0.0'.

    Returns True if duckdns responded with 'OK'.
    """
    params = {"domains": subdomain, "token": token}
    if ip is not None:
        params["ip"] = ip
    try:
        r = requests.get(DUCKDNS_UPDATE_URL, params=params, timeout=15)
        ok = r.text.strip() == "OK"
        # Token is not logged (not even partially, since it's short).
        if ok:
            print(f"[duckdns] {subdomain}.duckdns.org -> {ip or '<auto>'}")
        else:
            print(f"[duckdns] ERROR: '{r.text.strip()}' (token correct? "
                  f"subdomain '{subdomain}' created in duckdns account?)")
        return ok
    except Exception as e:
        print(f"[duckdns] Request failed: {e}")
        return False


def build_codeserver_env(*, domain: str, duckdns_token: str, llm_api_key: str,
                         codeserver_password: str,
                         model: str, ctx: int, parallel: int) -> dict:
    """Collect env entries for create_instance() in code-server mode.

    Vast.ai convention: docker -p mappings are passed as env keys
    ("-p HOST:CONTAINER" -> "1"); real container env as KEY -> VALUE.

    code-server's own PASSWORD env activates its login page with cookie
    auth. So we don't need BasicAuth in Caddy in front of it.
    """
    return {
        # Port mappings
        "-p 443:443": "1",     # Caddy HTTPS (the user-facing endpoint)
        "-p 8080:8080": "1",   # llama-server direct (opencode backward-compat)
        "-p 80:80": "1",       # ACME HTTP fallback (unused by Caddy, harmless)
        # Container env (read by /opt/onstart.sh)
        "DOMAIN": domain,
        "DUCKDNS_TOKEN": duckdns_token,
        "LLAMA_API_KEY": llm_api_key,
        "CODESERVER_PASSWORD": codeserver_password,
        "LLAMA_MODEL": model,
        "LLAMA_CTX": str(ctx),
        "LLAMA_PARALLEL": str(parallel),
        "HF_HOME": "/workspace/hf_cache",
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cmd_debug(args, vast: VastAI) -> None:
    """Try progressively stricter queries and show where hits fall to 0.

    Helps find filters that are broken or too strict - you see exactly which
    filter causes the hit-count cliff.
    """
    gpu_ram_q = round(args.min_vram * 0.93, 2)
    gpu_ram_real = int(args.min_vram * 1000 * 0.95)

    # Incremental stages: each adds exactly one filter to the previous.
    # That way you immediately see which filter kills the hits.
    incr = [
        ("num_gpus=1",                        "baseline (num_gpus=1)"),
        ("rentable=true rented=false",        "+ rentable/rented"),
        ("verified=true",                     "+ verified"),
        (f"dph_total<={args.max_price}",      f"+ dph_total<={args.max_price}"),
        (f"gpu_ram>={gpu_ram_q}",             f"+ gpu_ram>={gpu_ram_q} (GB!)"),
        (f"inet_down_cost<={args.max_inet_cost}",
            f"+ inet_down_cost<={args.max_inet_cost}"),
        (f"reliability>={args.min_reliability}",
            f"+ reliability>={args.min_reliability}"),
        (f"inet_down>={args.min_inet_down}",
            f"+ inet_down>={args.min_inet_down}"),
        ("direct_port_count>=2",              "+ direct_port_count>=2"),
    ]
    if args.region:
        incr.append((f"geolocation={args.region}",
                     f"+ geolocation={args.region}"))

    stages = []
    accumulated = []
    for filt, label in incr:
        accumulated.append(filt)
        stages.append((label, " ".join(accumulated)))

    print("=" * 72)
    print(f"DEBUG: min_vram={args.min_vram}GB max_price=${args.max_price}/h "
          f"min_rel={args.min_reliability} cuda_min={args.cuda_min} "
          f"max_inet_cost=${args.max_inet_cost}/GB")
    print(f"  query gpu_ram threshold: {gpu_ram_q} GB  (for query field, in GB!)")
    print(f"  post  gpu_ram threshold: {gpu_ram_real} MB (for response field, in MB)")
    print("=" * 72)

    last_count = None
    for label, query in stages:
        try:
            result = vast.search_offers(query=query,
                                         order="dlperf_per_dphtotal-",
                                         limit="100")
        except Exception as e:
            print(f"  [{label:<40}] ERROR: {e}")
            continue
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                print(f"  [{label:<40}] unparsable response: {result[:80]}")
                continue
        offers = result.get("offers", []) if isinstance(result, dict) else result
        n = len(offers)
        marker = ""
        if last_count is not None and n == 0 and last_count > 0:
            marker = "  <-- HITS DROP TO 0 HERE!"
        elif last_count is not None and n < last_count // 2 and last_count > 5:
            marker = f"  (halving: {last_count} -> {n})"
        print(f"  [{label:<40}] {n:>4} hits{marker}")
        last_count = n

    print()
    print("Additional Python-side filters (in the real run):")
    print(f"  - cuda_max_good >= {args.cuda_min} and not in {CUDA_BLACKLIST}")
    print(f"  - gpu_ram >= {gpu_ram_real} MB (response field)")
    print()
    print("Full run with verbose output:")
    print()
    offers = find_best_offers(vast, min_vram_gb=args.min_vram,
                              max_dph=args.max_price,
                              min_reliability=args.min_reliability,
                              min_inet_down=args.min_inet_down,
                              max_inet_cost=args.max_inet_cost,
                              region=args.region, cuda_min=args.cuda_min,
                              verbose=True)
    if not offers:
        print("NO offer found in the real run - look above to see where it fails.")


def cmd_destroy(vast: VastAI, instance_id: int,
                duckdns_token: str | None = None) -> None:
    """Destroy the instance and (optionally) clean up the duckdns entry.

    We read the instance env before destroying to extract DOMAIN if it was
    set in full-stack mode. If DOMAIN is found + duckdns_token is present,
    the A-record is set to 0.0.0.0 so no dead pointer is left behind.
    """
    domain = None
    try:
        inst_raw = vast.show_instance(id=instance_id)
        if isinstance(inst_raw, str):
            inst_raw = json.loads(inst_raw)
        inst = inst_raw.get("instances", inst_raw)
        if isinstance(inst, list):
            inst = inst[0] if inst else {}
        env = inst.get("extra_env") or inst.get("env") or {}
        # vast stores env partially as list of [key, value] pairs
        if isinstance(env, list):
            env = {k: v for pair in env if len(pair) == 2 for k, v in [pair]}
        domain = env.get("DOMAIN") if isinstance(env, dict) else None
    except Exception as e:
        print(f"[destroy] WARN: could not read instance env: {e}")

    print(f"[destroy] Deleting instance {instance_id}...")
    print(vast.destroy_instance(id=instance_id))

    if domain and duckdns_token:
        # 'mybox.duckdns.org' -> subdomain 'mybox'
        subdomain = domain.split(".duckdns.org")[0] if domain.endswith(".duckdns.org") else domain
        print(f"[destroy] Cleaning up duckdns: {domain} -> 0.0.0.0")
        update_duckdns(subdomain, duckdns_token, ip="0.0.0.0")
    elif domain and not duckdns_token:
        print(f"[destroy] NOTE: {domain} still points to the dead IP. "
              f"Clean up manually, or re-run destroy with --duckdns-token.")


def cmd_list(vast: VastAI) -> None:
    result = vast.show_instances()
    if isinstance(result, str):
        result = json.loads(result)
    instances = result.get("instances", []) if isinstance(result, dict) else result
    if not instances:
        print("No active instances.")
        return
    for i in instances:
        print(f"  id={i.get('id')} status={i.get('actual_status')} "
              f"gpu={i.get('gpu_name')} ${i.get('dph_total', 0):.3f}/h "
              f"label={i.get('label', '-')}")


def cmd_ssh(vast: VastAI, instance_id: int) -> None:
    """Attach an SSH key to the instance and exec into an interactive
    SSH session. Convenience for debugging (reading logs, checking model
    download, inspecting Caddy/llama-server).

    Reuses the gpu_summon_smoketest key. If the key doesn't exist yet,
    it's created here. attach_ssh is idempotent (the 'already attached'
    failure mode happens and is OK).

    os.execvp replaces the Python process with ssh so the user gets a
    real interactive shell (TTY, signals, ...) instead of a
    subprocess.run wrapper.
    """
    try:
        inst_raw = vast.show_instance(id=instance_id)
        if isinstance(inst_raw, str):
            inst_raw = json.loads(inst_raw)
        inst = inst_raw.get("instances", inst_raw)
        if isinstance(inst, list):
            inst = inst[0] if inst else {}
    except Exception as e:
        print(f"ERROR: could not read instance {instance_id}: {e}")
        sys.exit(2)

    ip = inst.get("public_ipaddr") or inst.get("ssh_host")
    ssh_port = get_ssh_port(inst)
    status = inst.get("actual_status") or inst.get("cur_state") or "?"
    if not ip or not ssh_port:
        print(f"ERROR: no SSH endpoint for instance {instance_id} "
              f"(status: {status}). Maybe not booted yet?")
        sys.exit(2)

    pub, priv = ensure_smoketest_keypair()

    try:
        r = vast.attach_ssh(instance_id=instance_id, ssh_key=pub)
        if isinstance(r, str):
            r = json.loads(r)
        if (not r.get("success") and
                "already" not in str(r.get("msg", "")).lower()):
            print(f"[ssh]  WARN attach_ssh: {r}")
    except Exception as e:
        print(f"[ssh]  WARN: attach_ssh failed ({e}); "
              f"trying to connect anyway...")

    print(f"[ssh]  ssh -i {priv} -p {ssh_port} root@{ip}")
    print()
    # execvp replaces this Python process with ssh -- real interactive
    # shell with TTY, signal handling etc.
    os.execvp("ssh", [
        "ssh",
        "-i", priv,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_port),
        f"root@{ip}",
    ])


def cmd_launch(args, vast: VastAI) -> None:
    # code-server mode: forces backend=codeserver, validates required inputs,
    # generates the workspace password. Must happen BEFORE the backend lookup.
    code_domain = None
    code_env = None
    code_password = None
    if args.with_codeserver:
        if not args.code_domain:
            print("ERROR: --with-codeserver requires --code-domain SUBDOMAIN")
            sys.exit(1)
        if not args.duckdns_token:
            print("ERROR: --with-codeserver requires DUCKDNS_TOKEN env var "
                  "or --duckdns-token TOKEN")
            sys.exit(1)
        if "." in args.code_domain:
            print("ERROR: --code-domain is just the subdomain "
                  "(e.g. 'mybox', NOT 'mybox.duckdns.org')")
            sys.exit(1)
        args.backend = "codeserver"
        code_domain = f"{args.code_domain}.duckdns.org"
        if args.disk < BACKENDS["codeserver"]["disk_gb_min"]:
            print(f"[code]   --disk {args.disk} too small, "
                  f"raising to {BACKENDS['codeserver']['disk_gb_min']}")
            args.disk = BACKENDS["codeserver"]["disk_gb_min"]

    backend_cfg = BACKENDS[args.backend]
    model = args.model or backend_cfg["default_model"]

    # Generate API key if not provided (or explicitly disabled).
    # Default: AUTO-GENERATE so the endpoint isn't open on the net.
    if args.llm_api_key is None:
        api_key = secrets.token_urlsafe(24)
        print(f"[auth]   No --llm-api-key given - generated: {api_key}")
    elif args.llm_api_key == "":
        api_key = ""
        print("[auth]   --llm-api-key='' -> no auth, endpoint is OPEN!")
    else:
        api_key = args.llm_api_key
        print(f"[auth]   Using --llm-api-key: {api_key[:8]}...")

    # code-server workspace password (only with --with-codeserver).
    if args.with_codeserver:
        if args.code_password is None:
            code_password = secrets.token_urlsafe(18)
            print(f"[code]   Workspace pass auto-generated: {code_password}")
        elif args.code_password == "":
            print("ERROR: empty code-server password - that would make the IDE "
                  "publicly reachable. Please set a password or leave the "
                  "default auto-gen.")
            sys.exit(1)
        else:
            code_password = args.code_password
            print(f"[code]   Using --code-password: {code_password[:4]}...")
        code_env = build_codeserver_env(
            domain=code_domain,
            duckdns_token=args.duckdns_token,
            llm_api_key=api_key,
            codeserver_password=code_password,
            model=model,
            ctx=args.num_ctx,
            parallel=args.parallel,
        )

    offers = find_best_offers(
        vast,
        min_vram_gb=args.min_vram,
        max_dph=args.max_price,
        min_reliability=args.min_reliability,
        min_inet_down=args.min_inet_down,
        max_inet_cost=args.max_inet_cost,
        region=args.region,
        cuda_min=args.cuda_min,
    )
    if not offers:
        sys.exit(1)

    if args.dry_run:
        o = offers[0]
        print(f"\n[dry-run] Primary pick: id={o['id']} {o['gpu_name']} "
              f"${o['dph_total']:.3f}/h - aborting.")
        if args.min_real_mbps > 0:
            print(f"[dry-run] (smoketest against {args.min_real_mbps} Mbps "
                  f"would run after container start.)")
        return

    # Prepare SSH key for smoketest if enabled
    ssh_pub = None
    ssh_priv = None
    if args.min_real_mbps > 0:
        ssh_pub, ssh_priv = ensure_smoketest_keypair()

    instance_id = None
    inst = None
    chosen_offer = None

    for attempt, offer in enumerate(offers, 1):
        print(f"\n[try]   Attempt {attempt}/{len(offers)}: "
              f"offer {offer['id']} ({offer['gpu_name']}, "
              f"${offer['dph_total']:.3f}/h, "
              f"inet_adv={offer.get('inet_down', 0):.0f}Mbps)")

        try:
            instance_id = create_instance(
                vast, offer, args.backend, model,
                disk_gb=args.disk, ctx=args.num_ctx, parallel=args.parallel,
                label=args.label, api_key=api_key,
                template_hash=args.template_hash,
                extra_env=code_env,
                image_override=args.code_image if args.with_codeserver else None,
            )
        except Exception as e:
            print(f"[try]   create_instance failed: {e}")
            instance_id = None
            continue

        try:
            inst = wait_until_running(vast, instance_id,
                                       timeout_sec=args.timeout)
        except Exception as e:
            print(f"[try]   Container start failed: {e}")
            try:
                vast.destroy_instance(id=instance_id)
            except Exception:
                pass
            instance_id = None
            continue

        # With --with-codeserver: vast.ai hosts almost always have port 443
        # already taken (their own web proxy). We accept the random port that
        # vast gives us and write it into all URLs. We used to try up to 3
        # offers - not worth it, just costs time and money.
        if args.with_codeserver:
            _, host_port = get_endpoint(inst, args.backend)
            if host_port != 443:
                print(f"[code]   host_port={host_port} (vast host has 443 "
                      f"taken) - URLs will be tagged with ':{host_port}'.")

        if args.min_real_mbps > 0:
            ip, _port = get_endpoint(inst, args.backend)
            ssh_port = get_ssh_port(inst)
            if not ssh_port:
                print("[probe] WARN: no SSH port mapped - skipping smoketest")
            else:
                mbps = smoketest_hf_bandwidth(
                    vast, instance_id, ip, ssh_port, ssh_pub, ssh_priv)
                print(f"[probe] Real HF->host bandwidth: {mbps:.1f} Mbps "
                      f"(threshold: {args.min_real_mbps:.0f} Mbps)")
                if mbps < args.min_real_mbps:
                    print("[probe] Too slow - destroying and trying "
                          "the next offer.")
                    try:
                        vast.destroy_instance(id=instance_id)
                    except Exception:
                        pass
                    instance_id = None
                    continue

        chosen_offer = offer
        break

    if chosen_offer is None or instance_id is None:
        print("\nERROR: No offer passed the bandwidth smoketest.")
        print("        Tip: lower --min-real-mbps, raise --min-inet-down, "
              "or retry at a different time of day.")
        sys.exit(2)

    try:
        ip, port = get_endpoint(inst, args.backend)
        print(f"\n[ready] Instance running on {ip}:{port}")

        if args.with_codeserver:
            # Actual host port mapped to container 443: 443 if the vast host
            # was lucky, otherwise random. We build all user-facing URLs
            # with a ":PORT" suffix when needed.
            port_suffix = "" if port == 443 else f":{port}"
            ide_url = f"https://code.{code_domain}{port_suffix}"
            llm_url_base = f"https://llm.{code_domain}{port_suffix}"

            # Set duckdns A-record to the new vast IP BEFORE Caddy attempts
            # an ACME DNS-01 challenge (otherwise the first cert attempt
            # fails). The TXT record comes from Caddy itself - we only set
            # the A-record.
            print(f"[code]   Setting duckdns: {code_domain} -> {ip}")
            ok = update_duckdns(args.code_domain, args.duckdns_token, ip=ip)
            if not ok:
                print("[code]   WARN: duckdns update failed - "
                      "Caddy ACME cert will probably not work.")

            print("[ready] Waiting for llama-server (5-15 min model download)...")
            llm_url = f"{llm_url_base}{backend_cfg['api_path']}/models"
            llm_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            wait_until_ready(llm_url, headers=llm_headers,
                             timeout_sec=args.model_timeout, label="llm")
            print("[ready] Waiting for code-server (Caddy ACME + login page)...")
            wait_until_ready(f"{ide_url}/",
                             timeout_sec=300, label="code")
        else:
            url = backend_cfg["ready_check"].format(host=ip, port=port)
            print("[ready] Waiting for model download and startup "
                  "(can take 5-15 min)...")
            wait_until_ready(url, timeout_sec=args.model_timeout)
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Tip: python {sys.argv[0]} --destroy {instance_id}")
        sys.exit(2)

    if args.write_config:
        # In codeserver mode also pass the host_port so the opencode
        # config contains the correct URL port (if != 443).
        write_opencode_config(ip, port, model, args.backend, api_key=api_key,
                              code_domain=code_domain,
                              code_port=(port if args.with_codeserver else None))

    # With --with-codeserver: AGENTS.md on the box has a
    # <HOST_PORT_SUFFIX> placeholder that we now replace with the real
    # port. Done via SSH (smoketest key is reused or created). Best-
    # effort - on SSH failure we print a hint instead of crashing.
    if args.with_codeserver:
        port_suffix = "" if port == 443 else f":{port}"
        ssh_port_for_inst = get_ssh_port(inst)
        if ssh_port_for_inst:
            try:
                pub, priv = ensure_smoketest_keypair()
                r = vast.attach_ssh(instance_id=instance_id, ssh_key=pub)
                if isinstance(r, str):
                    r = json.loads(r)
                # already-attached is also OK, only a complete failure is bad
                # Two SSH-side fixups: AGENTS.md placeholder + restart
                # code-server with the real VSCODE_PROXY_URI (so the Ports
                # tab in VS Code shows usable subdomain URLs incl. port).
                fixup_cmd = (
                    f"sed -i 's|<HOST_PORT_SUFFIX>|{port_suffix}|g' "
                    f"/workspace/projects/AGENTS.md && "
                    f"/opt/restart-codeserver.sh {port}"
                )
                subprocess.run([
                    "ssh",
                    "-i", priv,
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-o", "ConnectTimeout=8",
                    "-p", str(ssh_port_for_inst),
                    f"root@{ip}",
                    fixup_cmd,
                ], check=False, timeout=30, capture_output=True)
                print(f"[code]   AGENTS.md + code-server VSCODE_PROXY_URI "
                      f"set to host_port={port}.")
            except Exception as e:
                print(f"[code]   WARN: could not patch AGENTS.md: {e}")
                if port != 443:
                    print("[code]   Manually in the browser terminal:")
                    print(f"      sudo sed -i 's|<HOST_PORT_SUFFIX>|:{port}|g' "
                          "/workspace/projects/AGENTS.md")

    print("\n" + "="*60)
    if args.with_codeserver:
        port_suffix = "" if port == 443 else f":{port}"
        ide_url = f"https://code.{code_domain}{port_suffix}"
        llm_url_base = f"https://llm.{code_domain}{port_suffix}"
        print("DONE. AI dev environment is running.")
        print()
        print(f"  Browser IDE:   {ide_url}")
        print(f"  LLM endpoint:  {llm_url_base}{backend_cfg['api_path']}")
        if api_key:
            print(f"  LLM bearer:    {api_key}")
        print(f"  IDE login:     {code_password}   (code-server password)")
        if port != 443:
            print()
            print(f"  NOTE: vast host has 443 taken, using port {port}. "
                  f"Save your bookmark with ':{port}'.")
        print()
        print("Test:")
        if api_key:
            print(f"  curl -H 'Authorization: Bearer {api_key}' \\")
            print(f"       {llm_url_base}{backend_cfg['api_path']}/models")
        print(f"  Browser: {ide_url}")
        print()
        print("Start opencode (config points at the TLS endpoint):")
        print("  opencode")
    else:
        print("DONE. Endpoint:")
        print(f"  http://{ip}:{port}{backend_cfg['api_path']}")
        if api_key:
            print(f"  API key: {api_key}")
        print()
        print("Test:")
        if api_key:
            print(f"  curl -H 'Authorization: Bearer {api_key}' \\")
            print(f"       http://{ip}:{port}{backend_cfg['api_path']}/models")
        else:
            print(f"  curl http://{ip}:{port}{backend_cfg['api_path']}/models")
        print()
        print("Start opencode (config + auth have been written):")
        print("  opencode")
    print()
    print("IMPORTANT - clean up when done (otherwise the meter keeps running!):")
    if args.with_codeserver:
        print(f"  python {sys.argv[0]} --destroy {instance_id} \\")
        print("      --duckdns-token $DUCKDNS_TOKEN  # also cleans up DNS")
    else:
        print(f"  python {sys.argv[0]} --destroy {instance_id}")
    print("="*60)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--api-key", default=os.environ.get("VAST_API_KEY"))

    action = p.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true")
    action.add_argument("--destroy", type=int, metavar="ID")
    action.add_argument("--ssh", type=int, metavar="ID",
                        help="SSH login to a running instance. Creates "
                             "(if needed) the gpu-summon-smoketest key, "
                             "attaches it to the instance, and exec's into "
                             "an interactive shell. Handy for reading logs "
                             "or debugging a running box.")
    action.add_argument("--debug", action="store_true",
                        help="Try progressively stricter queries and show "
                             "where the hit count falls to 0.")

    p.add_argument("--backend", choices=list(BACKENDS.keys()), default="llamacpp",
                   help="Inference engine (default: llamacpp - recommended for Qwen3.6+)")
    p.add_argument("--model", default=None,
                   help="Model spec. Default depends on the backend.")
    p.add_argument("--min-vram", type=int, default=32)
    p.add_argument("--max-price", type=float, default=0.50)
    p.add_argument("--min-reliability", type=float, default=0.95,
                   help="Min reliability (default: 0.95). 0.98+ is very strict.")
    p.add_argument("--max-inet-cost", type=float, default=0.01,
                   help="Max download bandwidth cost in $/GB (default: 0.01). "
                        "On a ~25GB model pull that's worst-case $0.25/launch. "
                        "Values above 0.05 are common but expensive - a single "
                        "pull there costs over $1.")
    p.add_argument("--min-inet-down", type=int, default=200,
                   help="Min advertised download bandwidth in Mbps (default 200). "
                        "A floor for the filter - hosts often advertise more "
                        "than they deliver, so not very meaningful. For a real "
                        "test use --min-real-mbps.")
    p.add_argument("--min-real-mbps", type=float, default=0,
                   help="Real bandwidth smoketest at launch, in Mbps. "
                        "0 = disabled (default). When set: after container "
                        "start, real HF->host throughput is measured "
                        "(file-size growth in cache, 15s sample), and on "
                        "undershoot the instance is destroyed and the next "
                        "top-5 offer is tried. Catches hosts that advertise "
                        "200 Mbps inet_down but only deliver 80. Sensible "
                        "values: 100 (loose), 200 (medium), 400 (strict - "
                        "HF has to cooperate).")
    p.add_argument("--cuda-min", type=float, default=CUDA_MIN_GOOD,
                   help=f"Min CUDA version (default: {CUDA_MIN_GOOD}). "
                        f"13.2 is always excluded (gibberish bug).")
    p.add_argument("--region", default=None)
    p.add_argument("--disk", type=int, default=80)
    p.add_argument("--num-ctx", type=int, default=65536,
                   help="TOTAL context window (default: 65536). Split across "
                        "--parallel slots - with parallel=4 that's 16k per "
                        "user. With q8 KV cache, Qwen3.6 fits 128k+ total "
                        "on 32GB.")
    p.add_argument("--parallel", type=int, default=4,
                   help="Number of parallel inference slots (default: 4). "
                        "Higher = more concurrent users, but less ctx per "
                        "user and lower tok/s per user. Rule of thumb for "
                        "a 32GB GPU: 4 for solo/pair, 8 for a small "
                        "workshop group.")
    p.add_argument("--workshop-mode", action="store_true",
                   help="Shortcut: --parallel 8 --num-ctx 131072 "
                        "(= 16k per user across 8 slots). Optimized for "
                        "~5-10 concurrent users. Drops TTFT at 8 "
                        "concurrent from ~5s to ~1s, costs ~40%% in "
                        "per-user speed. On a 32GB GPU, ~6GB VRAM remain "
                        "as headroom.")
    p.add_argument("--solo-mode", action="store_true",
                   help="Shortcut: --parallel 2 --num-ctx 131072 "
                        "(= 64k per user across 2 slots). For solo work "
                        "with long conversations or large code files. "
                        "Full single-user speed (~53 tok/s), second slot "
                        "as headroom for tool calls / background requests "
                        "without queuing.")
    p.add_argument("--llm-api-key",
                   dest="llm_api_key",
                   default=os.environ.get("GPU_SUMMON_LLM_API_KEY"),
                   help="Bearer token for llama-server. Default: a 32-char "
                        "random token is generated. Empty string '' = no "
                        "auth (endpoint OPEN on the net). Settable via env "
                        "var GPU_SUMMON_LLM_API_KEY (e.g. for reusable keys "
                        "across multiple launches). Not to be confused with "
                        "--api-key (which is the VAST_API_KEY).")
    p.add_argument("--template-hash",
                   default=os.environ.get("GPU_SUMMON_TEMPLATE_HASH"),
                   help="Vast.ai template hash. Instead of rendering "
                        "onstart_cmd ourselves the template is used - image "
                        "+ onstart + runtype come from the template, the "
                        "script only injects mode env (LLAMA_PARALLEL/CTX/"
                        "MODEL). Settable via env var GPU_SUMMON_TEMPLATE_HASH.")

    # ----- code-server mode (--with-codeserver) -----
    p.add_argument("--with-codeserver", action="store_true",
                   help="Browser-based AI dev environment: llama-server PLUS "
                        "code-server (browser VS Code) on the same vast.ai "
                        "machine, behind Caddy with wildcard TLS via duckdns "
                        "+ Let's Encrypt. Requires --code-domain and "
                        "DUCKDNS_TOKEN. Uses the image from --code-image "
                        "(default: gpu-summon-codeserver), which must be "
                        "built beforehand via codeserver/build-and-push.sh "
                        "(or the GitHub Actions workflow).")
    p.add_argument("--code-domain", default=None,
                   help="Subdomain under duckdns.org, e.g. 'mybox' for "
                        "mybox.duckdns.org. Required with --with-codeserver. "
                        "The wildcard cert also covers llm.mybox.duckdns.org "
                        "(direct LLM access for opencode).")
    p.add_argument("--duckdns-token",
                   default=os.environ.get("DUCKDNS_TOKEN"),
                   help="duckdns.org API token (from https://www.duckdns.org/). "
                        "Settable via env var DUCKDNS_TOKEN. Needed for the "
                        "DNS update at launch and the ACME DNS-01 challenge "
                        "(wildcard cert).")
    p.add_argument("--code-image",
                   default="ghcr.io/dg1001/gpu-summon-codeserver:latest",
                   help="Docker image for code-server mode. Default: "
                        "ghcr.io/dg1001/gpu-summon-codeserver:latest. Build "
                        "your own via codeserver/build-and-push.sh or the "
                        "GitHub Actions workflow.")
    p.add_argument("--code-password",
                   default=os.environ.get("CODESERVER_PASSWORD"),
                   help="Workspace password for code-server's login page. "
                        "Default: 24-char auto-generated. Settable via env "
                        "var CODESERVER_PASSWORD. Empty string '' is "
                        "rejected - without a password the IDE would be public.")

    p.add_argument("--label", default="opencode-llm")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--model-timeout", type=int, default=1800)
    p.add_argument("--write-config", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.workshop_mode and args.solo_mode:
        print("ERROR: --workshop-mode and --solo-mode are mutually exclusive")
        sys.exit(1)
    if args.workshop_mode:
        args.parallel = 8
        args.num_ctx = 131072  # = 16k per slot, ~4.3 GB KV cache
    if args.solo_mode:
        args.parallel = 2
        args.num_ctx = 131072  # = 64k per slot, ~4.3 GB KV cache

    if not args.api_key:
        print("ERROR: VAST_API_KEY not set.")
        print("Get one at https://cloud.vast.ai/manage-keys/")
        sys.exit(1)

    vast = VastAI(api_key=args.api_key)

    if args.list:
        cmd_list(vast)
    elif args.destroy:
        cmd_destroy(vast, args.destroy, duckdns_token=args.duckdns_token)
    elif args.ssh:
        cmd_ssh(vast, args.ssh)
    elif args.debug:
        cmd_debug(args, vast)
    else:
        cmd_launch(args, vast)


if __name__ == "__main__":
    main()