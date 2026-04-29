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
    print("FEHLER: vastai SDK nicht installiert. Bitte: pip install vastai")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Backend-Definitionen
# -----------------------------------------------------------------------------

# Stand April 2026 - CUDA 13.2 hat einen bekannten Bug der bei vielen Modellen
# (insbesondere Qwen3.6) Gibberish-Output produziert. Wir filtern das aktiv aus.
CUDA_BLACKLIST = ["13.2"]
CUDA_MIN_GOOD = 12.0      # niedriger geht aber FP8/Flash-Attn will 12.0+

BACKENDS = {
    "llamacpp": {
        # Offizielles llama.cpp CUDA Server Image
        "image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
        "exposed_port": 8080,
        "api_path": "/v1",
        "default_model": "unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL",
        # Wichtig: --jinja fuer tool calling (opencode), --chat-template-kwargs
        # fuer Qwen3.6 Preserved Thinking, plus Sampling-Defaults von Unsloth
        # ctx-size ist GESAMT-Budget, nicht pro-Slot. Bei parallel=N kriegt
        # jeder Slot ctx-size/N. Default 65536 + parallel=4 -> 16k pro User,
        # reicht fuer normale Chat-Konversationen.
        # --api-key sichert das Endpoint per Bearer-Token. Ohne Key ist
        # llama-server PUBLIC erreichbar - jeder mit IP:PORT kann inferieren
        # auf deine Rechnung. {API_KEY} wird vom Skript injected.
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
}


# -----------------------------------------------------------------------------
# Offer-Suche
# -----------------------------------------------------------------------------

def _run_search(vast: VastAI, query: str, verbose: bool = True,
                cuda_min: float = 12.0) -> list:
    """Helper: fuehrt Query aus und gibt gefilterte Offers-Liste zurueck.

    CUDA-Filterung passiert hier Python-side, NICHT in der Query.
    Grund: vast.ai vergleicht cuda_max_good in der Query teils lexikalisch
    (String), wodurch '12.10' faelschlich kleiner als '12.4' ist und
    legitime Hosts rausfallen. Floats vergleichen sich Python-side korrekt.
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
        print(f"[search] FEHLER bei search_offers: {e}")
        return []
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print(f"[search] Konnte Antwort nicht parsen: {result[:200]}")
            return []
    raw = result.get("offers", []) if isinstance(result, dict) else result

    if verbose and raw:
        print(f"[search]   -> {len(raw)} rohe Treffer von der API")

    # CUDA-Filterung Python-side: min Version + 13.2 Blacklist
    def cuda_ok(o):
        try:
            cv = float(o.get("cuda_max_good", 0))
        except (ValueError, TypeError):
            return False
        return cv >= cuda_min and str(o.get("cuda_max_good")) not in CUDA_BLACKLIST

    filtered = [o for o in raw if cuda_ok(o)]
    if verbose and len(raw) != len(filtered):
        print(f"[search]   -> {len(filtered)} nach CUDA-Filter "
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
    Sucht passende GPU mit Filter "preisguenstig aber stabil".

    Stolperfallen die hier umgangen werden:
    - 'reliability' im Query, 'reliability2' im Response (vast.ai Inkonsistenz)
    - gpu_ram in der QUERY ist GB (Float erlaubt), im RESPONSE aber MB.
      Wer hier 30000 schreibt sucht nach Karten mit 30 TB VRAM -> 0 Treffer.
    - gpu_ram nominal vs reserved: 32GB-Karten melden oft 32510 statt 32768
    - cuda_max_good in Query vergleicht teils lexikalisch -> Python-side filtern
    - CUDA 13.2 produziert Gibberish bei Qwen3.6 -> blacklist
    """
    # Query: GB (Float), mit 7% Puffer fuer Karten die etwas weniger melden
    gpu_ram_query_threshold = round(min_vram_gb * 0.93, 2)
    # Post-Filter: MB, strikter (5% Puffer) gegen den Response
    gpu_ram_real_threshold = int(min_vram_gb * 1000 * 0.95)

    # inet_down_cost gehoert zu base: ist Kostenbegrenzung wie dph_total.
    # Manche Hosts verlangen $0.04/GB - bei 25 GB Modell-Pull schon $1.
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
        """Python-side: zu kleine GPU-RAM Karten rauswerfen."""
        return [o for o in offers
                if o.get("gpu_ram", 0) >= gpu_ram_real_threshold]

    def search(filters):
        query = " ".join(filters)
        return post_filter(_run_search(vast, query, verbose, cuda_min=cuda_min))

    # Versuch 1: alle Filter
    offers = search(base_filters + quality_filters)
    if verbose and offers:
        print(f"[search] Versuch 1: {len(offers)} Treffer nach allen Filtern")

    # Versuch 2: ohne Bandbreiten/Port-Filter
    if not offers:
        if verbose:
            print("[search] Keine Treffer - lockere inet_down/direct_port...")
        relaxed = [f for f in quality_filters
                   if not f.startswith(("inet_down", "direct_port_count"))]
        offers = search(base_filters + relaxed)

    # Versuch 3: nur Basis + verified
    if not offers:
        if verbose:
            print("[search] Immer noch nichts - nur verified+rentable...")
        offers = search(base_filters + ["verified=true"])

    # Versuch 4: alle Filter weg, nur Basis
    if not offers:
        if verbose:
            print("[search] Letzter Versuch - nur Basis-Filter...")
        offers = search(base_filters)

    if not offers:
        print("[search] Keine passenden Offers gefunden.")
        print("         Tipp: --max-price erhoehen, --min-vram senken, "
              "oder --region weglassen.")
        return []

    if verbose:
        print(f"[search] {len(offers)} Offers gefunden, Top 5:")
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

    # Aus Top-5 nach Reliability sortiert: erstes Element ist Erstwahl,
    # weitere sind Backup falls Smoketest fehlschlaegt.
    top = sorted(offers[:5], key=lambda o: (-o['reliability2'], o['dph_total']))
    if verbose:
        print("[search] Reihenfolge nach Reliability:")
        for i, o in enumerate(top, 1):
            print(f"   {i}. id={o['id']} {o['gpu_name']}  "
                  f"${o['dph_total']:.3f}/h  rel={o['reliability2']:.3f}  "
                  f"inet_adv={o.get('inet_down', 0):.0f}Mbps")
    return top


def find_best_offer(vast: VastAI, **kwargs) -> dict | None:
    """Backward-compat: erstes Element der find_best_offers Liste."""
    offers = find_best_offers(vast, **kwargs)
    return offers[0] if offers else None


# -----------------------------------------------------------------------------
# Instanz starten
# -----------------------------------------------------------------------------

def create_instance(vast: VastAI, offer: dict, backend: str, model: str, *,
                    disk_gb: int = 80, ctx: int = 65536, parallel: int = 4,
                    label: str = "opencode-llm",
                    api_key: str = "",
                    template_hash: str | None = None) -> int:
    """Erstellt die Instanz mit dem gewuenschten Backend-Image.

    api_key: Bearer-Token das llama-server akzeptiert. Leer = kein Auth
    (Endpoint public). Auto-generiert von cmd_launch wenn nicht gesetzt.
    template_hash: optional, Hash eines vast.ai Templates. Wenn gesetzt
    werden image + onstart + runtype aus dem Template genommen, und das
    Skript injected nur Mode-Env (LLAMA_PARALLEL/LLAMA_CTX/LLAMA_MODEL/
    LLAMA_API_KEY). Aktuell nur fuer llamacpp-Backend mit env-aware
    onstart unterstuetzt.
    """
    cfg = BACKENDS[backend]
    use_template = template_hash is not None

    if use_template and backend != "llamacpp":
        raise ValueError(
            f"--template-hash aktuell nur mit backend=llamacpp unterstuetzt, "
            f"nicht {backend}")

    print(f"[create] Backend: {backend}")
    print(f"[create] Modell: {model}")
    if backend == "llamacpp":
        print(f"[create] ctx={ctx} parallel={parallel} "
              f"({ctx//parallel} tokens pro Slot)")
    if use_template:
        print(f"[create] Template: {template_hash} "
              f"(image+onstart aus Template)")
    else:
        print(f"[create] Image: {cfg['image']}")
    print(f"[create] Erstelle Instanz auf Offer {offer['id']}...")

    env = {
        f"-p {cfg['exposed_port']}:{cfg['exposed_port']}": "1",
    }
    if backend == "llamacpp":
        env["HF_HOME"] = "/workspace/hf_cache"
        if use_template:
            # Template's onstart liest diese ENVs mit Defaults
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

    if use_template:
        # Template liefert image + onstart_cmd + runtype.
        # Env wird hier komplett gesetzt (vast merged NICHT mit Template-env,
        # es ersetzt). Deshalb -p, HF_HOME etc. mit reingenommen.
        result = vast.create_instance(
            id=offer["id"],
            disk=disk_gb,
            env=env,
            template_hash=template_hash,
            label=label,
        )
    else:
        # Self-contained Pfad: image + onstart_cmd selbst rendern
        if backend == "llamacpp":
            onstart = cfg["onstart_template"].format(
                MODEL=model, CTX=ctx, PARALLEL=parallel, API_KEY=api_key or "")
        else:
            onstart = cfg["onstart_template"].format(MODEL=model, CTX=ctx)
        # SDK akzeptiert ssh/direct NICHT als kwargs - der CLI uebersetzt
        # --ssh --direct intern zu runtype="ssh_direc ssh_proxy". Wer hier
        # ssh=True schreibt kassiert TypeError.
        result = vast.create_instance(
            id=offer["id"],
            image=cfg["image"],
            disk=disk_gb,
            env=env,
            onstart_cmd=onstart,
            runtype="ssh_direc ssh_proxy",
            label=label,
        )
    if isinstance(result, str):
        result = json.loads(result)
    if not result.get("success"):
        raise RuntimeError(f"create_instance fehlgeschlagen: {result}")

    instance_id = result["new_contract"]
    print(f"[create] Instance ID: {instance_id}")
    return instance_id


def wait_until_running(vast: VastAI, instance_id: int,
                       timeout_sec: int = 600) -> dict:
    """Pollt bis Instanz 'running' ist und Ports gemappt sind."""
    print(f"[wait]  Warte auf 'running' Status (max {timeout_sec}s)...")
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
                    f"Instanz im Fehlerzustand: {status}. "
                    f"Logs pruefen oder neue Offer probieren."
                )

            if status == "running" and inst.get("ports"):
                print(f"[wait]  Ports aktiv: {list(inst['ports'].keys())}")
                return inst

            time.sleep(5)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[wait]  WARN: {e}")
            time.sleep(5)

    raise TimeoutError(f"Instanz {instance_id} nicht in {timeout_sec}s ready")


def get_endpoint(inst: dict, backend: str) -> tuple[str, int]:
    """Liest IP und gemappten Port fuer das gewaehlte Backend."""
    ip = inst.get("public_ipaddr") or inst.get("ssh_host")
    ports = inst.get("ports") or {}
    internal = f"{BACKENDS[backend]['exposed_port']}/tcp"
    mapping = ports.get(internal)
    port = int(mapping[0]["HostPort"]) if mapping else None
    return ip, port


def wait_until_ready(host: str, port: int, backend: str,
                     timeout_sec: int = 1800) -> None:
    """
    Pollt die API bis das Modell tatsaechlich geladen ist.
    llama-server laedt das Modell beim ersten Start von HuggingFace -
    je nach Modellgroesse 5-15 Minuten.
    """
    url = BACKENDS[backend]["ready_check"].format(host=host, port=port)
    print(f"[ready] Pruefe {url}...")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                print(f"[ready] API antwortet (nach {int(time.time()-start)}s)")
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        elapsed = int(time.time() - start)
        if elapsed % 30 < 5:
            print(f"[ready] [{elapsed}s] noch nicht bereit, warte weiter...")
        time.sleep(5)
    raise TimeoutError(f"Modell nach {timeout_sec}s nicht bereit auf {url}")


# -----------------------------------------------------------------------------
# Bandwidth-Smoketest (echte HF->Host Geschwindigkeit messen)
# -----------------------------------------------------------------------------

def get_ssh_port(inst: dict) -> int | None:
    ports = inst.get("ports") or {}
    mapping = ports.get("22/tcp")
    return int(mapping[0]["HostPort"]) if mapping else None


def ensure_smoketest_keypair() -> tuple[str, str]:
    """Generiert ed25519-Keypair fuer Smoketest-SSH falls noch keins da.

    Eigener Key (~/.ssh/gpu_summon_smoketest), damit das nicht mit User-Keys
    kollidiert. Returns (public_key_string, private_key_path).
    """
    key_dir = Path.home() / ".ssh"
    key_dir.mkdir(parents=True, exist_ok=True)
    priv = key_dir / "gpu_summon_smoketest"
    pub = key_dir / "gpu_summon_smoketest.pub"
    if not priv.exists():
        print(f"[probe] Generiere SSH-Key fuer Smoketest: {priv}")
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
    """Misst echte HF->Host-Bandbreite waehrend des Modell-Downloads.

    Strategie: SSH rein, polle bis '.downloadInProgress' Cache-File existiert,
    sample File-Size-Wachstum ueber probe_seconds. Das misst genau das was
    am Ende zaehlt - nicht theoretische sondern effektive HF-Geschwindigkeit
    fuer diesen Host gerade jetzt.

    Return: gemessene Mbps (float). 0.0 = Probe fehlgeschlagen (egal ob
    SSH, Timeout, Parse - alles als 'unbrauchbar' behandelt).
    """
    try:
        r = vast.attach_ssh(instance_id=instance_id, ssh_key=ssh_key_pub)
        if isinstance(r, str):
            r = json.loads(r)
        # already-associated ist OK, alles andere ggf nicht.
        if not r.get("success") and \
           "already" not in str(r.get("msg", "")).lower():
            print(f"[probe] WARN attach_ssh: {r}")
    except Exception as e:
        print(f"[probe] attach_ssh Fehler: {e}")
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

    print(f"[probe] Warte auf SSH zu {ip}:{ssh_port}...")
    deadline = time.time() + 60
    while time.time() < deadline:
        r = subprocess.run(ssh_args + ["echo OK"],
                           capture_output=True, timeout=10, text=True)
        if r.returncode == 0:
            break
        time.sleep(3)
    else:
        print("[probe] SSH nicht erreichbar nach 60s")
        return 0.0

    print(f"[probe] Warte auf Start des HF-Downloads "
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
        print(f"[probe] Kein .downloadInProgress nach {wait_for_download_sec}s "
              f"- Download nicht gestartet, llama-server haengt?")
        return 0.0
    print(f"[probe] Download laeuft: {blob}")

    print(f"[probe] Sample {probe_seconds}s File-Size-Wachstum...")
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
        print(f"[probe] Mess-SSH-Befehl Fehler: {r.stderr.strip()}")
        return 0.0
    try:
        s1_str, s2_str = r.stdout.strip().split()
        s1, s2 = int(s1_str), int(s2_str)
        bps = max(0, s2 - s1) / probe_seconds
        return bps * 8 / 1_000_000
    except Exception as e:
        print(f"[probe] Parse-Fehler: {e}, output={r.stdout!r}")
        return 0.0


# -----------------------------------------------------------------------------
# opencode Config
# -----------------------------------------------------------------------------

def write_opencode_config(host: str, port: int, model: str, backend: str,
                           api_key: str = "",
                           config_dir: Path | None = None) -> Path:
    """Schreibt opencode.json + auth.json fuer das remote Setup.

    api_key: das Bearer-Token das llama-server erwartet. Leer = kein Auth
    (Default-Dummy-Key wird geschrieben, llama-server akzeptiert dann eh
    alles).
    """
    if config_dir is None:
        config_dir = Path.home() / ".config" / "opencode"
    auth_dir = Path.home() / ".local" / "share" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    auth_dir.mkdir(parents=True, exist_ok=True)

    api_path = BACKENDS[backend]["api_path"]
    base_url = f"http://{host}:{port}{api_path}"
    provider_id = f"vast-{backend}"

    # Echten Modell-Namen vom Server holen damit der Key in opencode stimmt.
    # Wenn auth aktiv ist mit dem api-key versuchen.
    model_id = model
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(f"{base_url}/models", timeout=10, headers=headers)
        if r.ok:
            models_data = r.json().get("data", [])
            if models_data:
                model_id = models_data[0]["id"]
                print(f"[cfg]   Modell-ID vom Server: {model_id}")
    except Exception as e:
        print(f"[cfg]   WARN: konnte Modell-ID nicht abrufen, nutze {model}: {e}")

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"Vast.ai {backend} ({host})",
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
            print("[cfg]   WARN: existierende Config nicht parsbar, ueberschreibe")

    config_path.write_text(json.dumps(config, indent=2))
    print(f"[cfg]   {config_path}")

    auth_path = auth_dir / "auth.json"
    auth = {}
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text())
        except json.JSONDecodeError:
            pass
    # Wenn api_key gesetzt: opencode schickt das als Bearer; llama-server
    # validiert. Leer = der frueher genutzte Dummy-Wert reicht weil
    # llama-server ohne --api-key alles akzeptiert.
    auth[provider_id] = {"type": "api", "key": api_key or backend}
    auth_path.write_text(json.dumps(auth, indent=2))
    print(f"[cfg]   {auth_path}")

    return config_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cmd_debug(args, vast: VastAI) -> None:
    """Probiert progressiv strengere Queries und zeigt wo die Treffer auf 0 fallen.

    Hilft Filter zu finden die kaputt sind oder zu strikt - du siehst genau
    welcher Filter den Trefferzahl-Cliff verursacht.
    """
    gpu_ram_q = round(args.min_vram * 0.93, 2)
    gpu_ram_real = int(args.min_vram * 1000 * 0.95)

    # Inkrementelle Stages: jede addiert genau einen Filter zur vorherigen.
    # So sieht man sofort welcher Filter die Treffer killt.
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
    print(f"  query gpu_ram threshold: {gpu_ram_q} GB  (fuer Query-Field, in GB!)")
    print(f"  post  gpu_ram threshold: {gpu_ram_real} MB (fuer Response-Field, in MB)")
    print("=" * 72)

    last_count = None
    for label, query in stages:
        try:
            result = vast.search_offers(query=query,
                                         order="dlperf_per_dphtotal-",
                                         limit="100")
        except Exception as e:
            print(f"  [{label:<40}] FEHLER: {e}")
            continue
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                print(f"  [{label:<40}] unparsbare Antwort: {result[:80]}")
                continue
        offers = result.get("offers", []) if isinstance(result, dict) else result
        n = len(offers)
        marker = ""
        if last_count is not None and n == 0 and last_count > 0:
            marker = "  <-- HIER bricht's ein!"
        elif last_count is not None and n < last_count // 2 and last_count > 5:
            marker = f"  (Halbierung: {last_count} -> {n})"
        print(f"  [{label:<40}] {n:>4} Treffer{marker}")
        last_count = n

    print()
    print("Zusaetzlich Python-side Filter (im echten Lauf):")
    print(f"  - cuda_max_good >= {args.cuda_min} und nicht in {CUDA_BLACKLIST}")
    print(f"  - gpu_ram >= {gpu_ram_real} MB (Response-Feld)")
    print()
    print("Kompletter Lauf mit verbose Ausgabe:")
    print()
    offers = find_best_offers(vast, min_vram_gb=args.min_vram,
                              max_dph=args.max_price,
                              min_reliability=args.min_reliability,
                              min_inet_down=args.min_inet_down,
                              max_inet_cost=args.max_inet_cost,
                              region=args.region, cuda_min=args.cuda_min,
                              verbose=True)
    if not offers:
        print("KEINE Offer gefunden im echten Lauf - oben sehen wo's haengt.")


def cmd_destroy(vast: VastAI, instance_id: int) -> None:
    print(f"[destroy] Loesche Instanz {instance_id}...")
    print(vast.destroy_instance(id=instance_id))


def cmd_list(vast: VastAI) -> None:
    result = vast.show_instances()
    if isinstance(result, str):
        result = json.loads(result)
    instances = result.get("instances", []) if isinstance(result, dict) else result
    if not instances:
        print("Keine aktiven Instanzen.")
        return
    for i in instances:
        print(f"  id={i.get('id')} status={i.get('actual_status')} "
              f"gpu={i.get('gpu_name')} ${i.get('dph_total', 0):.3f}/h "
              f"label={i.get('label', '-')}")


def cmd_launch(args, vast: VastAI) -> None:
    backend_cfg = BACKENDS[args.backend]
    model = args.model or backend_cfg["default_model"]

    # API-Key generieren falls nicht angegeben (oder explizit deaktiviert).
    # Default: AUTO-GENERATE damit der Endpoint nicht offen im Netz steht.
    if args.llm_api_key is None:
        api_key = secrets.token_urlsafe(24)
        print(f"[auth]   Kein --llm-api-key angegeben - generiert: {api_key}")
    elif args.llm_api_key == "":
        api_key = ""
        print("[auth]   --llm-api-key='' -> kein Auth, Endpoint ist OFFEN!")
    else:
        api_key = args.llm_api_key
        print(f"[auth]   Nutze --llm-api-key: {api_key[:8]}...")

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
        print(f"\n[dry-run] Erstwahl: id={o['id']} {o['gpu_name']} "
              f"${o['dph_total']:.3f}/h - Abbruch.")
        if args.min_real_mbps > 0:
            print(f"[dry-run] (Smoketest gegen {args.min_real_mbps} Mbps "
                  f"wuerde nach Container-Start laufen.)")
        return

    # SSH-Key fuer Smoketest vorbereiten falls aktiviert
    ssh_pub = None
    ssh_priv = None
    if args.min_real_mbps > 0:
        ssh_pub, ssh_priv = ensure_smoketest_keypair()

    instance_id = None
    inst = None
    chosen_offer = None

    for attempt, offer in enumerate(offers, 1):
        print(f"\n[try]   Versuch {attempt}/{len(offers)}: "
              f"Offer {offer['id']} ({offer['gpu_name']}, "
              f"${offer['dph_total']:.3f}/h, "
              f"inet_adv={offer.get('inet_down', 0):.0f}Mbps)")

        try:
            instance_id = create_instance(
                vast, offer, args.backend, model,
                disk_gb=args.disk, ctx=args.num_ctx, parallel=args.parallel,
                label=args.label, api_key=api_key,
                template_hash=args.template_hash,
            )
        except Exception as e:
            print(f"[try]   create_instance fehlgeschlagen: {e}")
            instance_id = None
            continue

        try:
            inst = wait_until_running(vast, instance_id,
                                       timeout_sec=args.timeout)
        except Exception as e:
            print(f"[try]   Container Start fehlgeschlagen: {e}")
            try:
                vast.destroy_instance(id=instance_id)
            except Exception:
                pass
            instance_id = None
            continue

        if args.min_real_mbps > 0:
            ip, _port = get_endpoint(inst, args.backend)
            ssh_port = get_ssh_port(inst)
            if not ssh_port:
                print("[probe] WARN: kein SSH-Port gemappt - skipping smoketest")
            else:
                mbps = smoketest_hf_bandwidth(
                    vast, instance_id, ip, ssh_port, ssh_pub, ssh_priv)
                print(f"[probe] Echte HF->Host Bandbreite: {mbps:.1f} Mbps "
                      f"(Threshold: {args.min_real_mbps:.0f} Mbps)")
                if mbps < args.min_real_mbps:
                    print("[probe] Zu langsam - zerstoere und probiere "
                          "naechsten Offer.")
                    try:
                        vast.destroy_instance(id=instance_id)
                    except Exception:
                        pass
                    instance_id = None
                    continue

        chosen_offer = offer
        break

    if chosen_offer is None or instance_id is None:
        print("\nFEHLER: Kein Offer hat den Bandwidth-Smoketest bestanden.")
        print("        Tipp: --min-real-mbps senken, --min-inet-down "
              "erhoehen, oder zu anderer Tageszeit retry.")
        sys.exit(2)

    try:
        ip, port = get_endpoint(inst, args.backend)
        print(f"\n[ready] Instance laeuft auf {ip}:{port}")
        print("[ready] Warte auf Modell-Download und Start "
              "(kann 5-15 Min dauern)...")
        wait_until_ready(ip, port, args.backend, timeout_sec=args.model_timeout)
    except Exception as e:
        print(f"\nFEHLER: {e}")
        print(f"Tipp: python {sys.argv[0]} --destroy {instance_id}")
        sys.exit(2)

    if args.write_config:
        write_opencode_config(ip, port, model, args.backend, api_key=api_key)

    print("\n" + "="*60)
    print("FERTIG. Endpoint:")
    print(f"  http://{ip}:{port}{backend_cfg['api_path']}")
    if api_key:
        print(f"  API-Key: {api_key}")
    print()
    print("Test:")
    if api_key:
        print(f"  curl -H 'Authorization: Bearer {api_key}' \\")
        print(f"       http://{ip}:{port}{backend_cfg['api_path']}/models")
    else:
        print(f"  curl http://{ip}:{port}{backend_cfg['api_path']}/models")
    print()
    print("opencode starten (Config + Auth sind geschrieben):")
    print("  opencode")
    print()
    print("WICHTIG - Aufraeumen wenn fertig (sonst laeuft die Stundenuhr!):")
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
    action.add_argument("--debug", action="store_true",
                        help="Probiert progressiv strengere Queries und zeigt "
                             "wo die Trefferzahl auf 0 faellt.")

    p.add_argument("--backend", choices=list(BACKENDS.keys()), default="llamacpp",
                   help="Inference Engine (default: llamacpp - empfohlen fuer Qwen3.6+)")
    p.add_argument("--model", default=None,
                   help="Modell-Spezifikation. Default je nach Backend.")
    p.add_argument("--min-vram", type=int, default=32)
    p.add_argument("--max-price", type=float, default=0.50)
    p.add_argument("--min-reliability", type=float, default=0.95,
                   help="Min reliability (default: 0.95). 0.98+ ist sehr strikt.")
    p.add_argument("--max-inet-cost", type=float, default=0.01,
                   help="Max Download-Bandbreitenkosten in $/GB (default: 0.01). "
                        "Bei ~25GB Modell-Pull macht das im worst case "
                        "$0.25/Launch. Werte ueber 0.05 sind ueblich aber "
                        "teuer - dort kostet ein Pull ueber $1.")
    p.add_argument("--min-inet-down", type=int, default=200,
                   help="Min beworbene Download-Bandbreite in Mbps (default 200). "
                        "Floor fuer den Filter - Hosts werben aber gerne mehr "
                        "als sie liefern, also wenig aussagekraeftig. Fuer "
                        "echten Test --min-real-mbps benutzen.")
    p.add_argument("--min-real-mbps", type=float, default=0,
                   help="Echter Bandbreiten-Smoketest beim Launch in Mbps. "
                        "0 = deaktiviert (default). Wenn gesetzt: nach Container-"
                        "Start wird die echte HF->Host-Geschwindigkeit gemessen "
                        "(File-Size-Wachstum im Cache, 15s sample), und bei "
                        "Unterschreiten wird die Instanz zerstoert und der "
                        "naechste Top-5-Offer probiert. Faengt Hosts ab die "
                        "200Mbps inet_down advertisen aber real nur 80 liefern. "
                        "Sinnvolle Werte: 100 (locker), 200 (mittel), 400 "
                        "(strikt - HF muss mitspielen).")
    p.add_argument("--cuda-min", type=float, default=CUDA_MIN_GOOD,
                   help=f"Min CUDA Version (default: {CUDA_MIN_GOOD}). "
                        f"13.2 wird immer ausgeschlossen (Gibberish-Bug).")
    p.add_argument("--region", default=None)
    p.add_argument("--disk", type=int, default=80)
    p.add_argument("--num-ctx", type=int, default=65536,
                   help="GESAMT-Context-Window (default: 65536). Wird auf "
                        "--parallel Slots aufgeteilt - bei parallel=4 also "
                        "16k pro User. Mit q8 KV-Cache passen bei Qwen3.6 "
                        "sogar 128k+ Gesamt auf 32GB.")
    p.add_argument("--parallel", type=int, default=4,
                   help="Anzahl paralleler Inferenz-Slots (default: 4). "
                        "Hoeher = mehr User gleichzeitig, aber weniger ctx "
                        "pro User und weniger tok/s pro User. Faustformel "
                        "fuer 32GB GPU: 4 fuer Solo/Paar, 8 fuer kleine "
                        "Workshop-Gruppe.")
    p.add_argument("--workshop-mode", action="store_true",
                   help="Shortcut: --parallel 8 --num-ctx 131072 "
                        "(= 16k pro User auf 8 Slots). Optimiert fuer "
                        "~5-10 gleichzeitige User. Setzt TTFT bei 8 "
                        "concurrent von ~5s auf ~1s, kostet ~40%% "
                        "pro-User-Speed. Auf 32GB GPU verbleiben ~6GB "
                        "VRAM Reserve.")
    p.add_argument("--solo-mode", action="store_true",
                   help="Shortcut: --parallel 2 --num-ctx 131072 "
                        "(= 64k pro User auf 2 Slots). Fuer Solo-Arbeit "
                        "mit langen Konversationen oder grossen "
                        "Code-Files. Voller Single-User-Speed (~53 tok/s), "
                        "zweiter Slot als Reserve fuer Tool-Calls / "
                        "Background-Anfragen ohne Queuing.")
    p.add_argument("--llm-api-key",
                   dest="llm_api_key",
                   default=os.environ.get("GPU_SUMMON_LLM_API_KEY"),
                   help="Bearer-Token fuer llama-server. Default: 32-char "
                        "Random-Token wird generiert. Leerstring '' = kein "
                        "Auth (Endpoint OFFEN im Netz). Setzbar via env-var "
                        "GPU_SUMMON_LLM_API_KEY (z.B. fuer wiederverwendbare "
                        "Keys ueber mehrere Launches hinweg). Nicht zu "
                        "verwechseln mit --api-key (das ist der VAST_API_KEY).")
    p.add_argument("--template-hash",
                   default=os.environ.get("GPU_SUMMON_TEMPLATE_HASH"),
                   help="Vast.ai Template-Hash. Statt onstart_cmd selbst zu "
                        "rendern wird das Template benutzt - image + onstart + "
                        "runtype kommen aus dem Template, nur Mode-Env "
                        "(LLAMA_PARALLEL/CTX/MODEL) wird vom Skript injected. "
                        "Setzbar via env-var GPU_SUMMON_TEMPLATE_HASH.")
    p.add_argument("--label", default="opencode-llm")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--model-timeout", type=int, default=1800)
    p.add_argument("--write-config", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.workshop_mode and args.solo_mode:
        print("FEHLER: --workshop-mode und --solo-mode schliessen sich aus")
        sys.exit(1)
    if args.workshop_mode:
        args.parallel = 8
        args.num_ctx = 131072  # = 16k pro Slot, ~4.3 GB KV-Cache
    if args.solo_mode:
        args.parallel = 2
        args.num_ctx = 131072  # = 64k pro Slot, ~4.3 GB KV-Cache

    if not args.api_key:
        print("FEHLER: VAST_API_KEY nicht gesetzt.")
        print("Hole dir einen unter https://cloud.vast.ai/manage-keys/")
        sys.exit(1)

    vast = VastAI(api_key=args.api_key)

    if args.list:
        cmd_list(vast)
    elif args.destroy:
        cmd_destroy(vast, args.destroy)
    elif args.debug:
        cmd_debug(args, vast)
    else:
        cmd_launch(args, vast)


if __name__ == "__main__":
    main()