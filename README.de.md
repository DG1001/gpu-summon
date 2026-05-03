# gpu-summon

Ein Kommando um einen self-hosted [llama.cpp](https://github.com/ggml-org/llama.cpp) Endpoint auf einer gemieteten Cloud-GPU zu beschwoeren und in den Editor zu verdrahten. Gebaut fuer Coding-Sessions wo Daten-Lokalitaet wichtig ist — dein Code geht nie an eine Cloud-LLM-API.

GPU-Marketplace-Anbindung: aktuell [vast.ai](https://vast.ai). Architektur ist so ausgelegt dass weitere Anbieter (RunPod, Lambda, Salad, …) ergaenzt werden koennen ohne dass die UX des Launchers sich aendert.

> Sprachen: [English](README.md) · **Deutsch** · [LESSONS.md](LESSONS.md) (Tiefgang)

## Was es macht

- Mietet eine Cloud-GPU und startet `llama-server` darauf mit einem GGUF-Modell deiner Wahl (Default: Unsloth Qwen3.6-27B Q5_K_XL)
- Sucht den Offer fuer dich anhand Kosten + Reliability + Bandbreite + CUDA-Version
- Optional: **misst die echte Download-Bandbreite** bevor commitet wird (gemietete Hosts werben oft 200 Mbps und liefern 80)
- Generiert defaultmaessig ein Bearer-Token — der Endpoint ist nicht offen im Netz
- Schreibt eine fertige [opencode](https://opencode.ai) Config — danach reicht `opencode`
- Aufraeumen mit einem Befehl, damit der Stundenzaehler nicht weiterlaeuft

## Warum?

Cloud-Coding-LLMs sind teuer und dein Code ist deins. Fuer Sessions wo Daten lokal bleiben sollen aber eine 32 GB GPU lokal nicht drin ist, ist eine 30 ¢/h Cloud-GPU mit self-hosted llama-server der Sweet Spot — und `gpu-summon` macht aus 30 Minuten CLI-Archaeologie ein Kommando.

## Quick Start

```bash
git clone https://github.com/<dein-user>/gpu-summon.git
cd gpu-summon
pip install vastai requests

export VAST_API_KEY=...   # von https://cloud.vast.ai/manage-keys/

# Solo-Coding: 64k Context pro User, secure-by-default
python summon.py --solo-mode

# In eine laufende Instanz reinschauen (Modell-Download-Fortschritt, Logs, ...)
python summon.py --ssh <id>

# Wenn fertig
python summon.py --list
python summon.py --destroy <id>
```

### Innen in der Instanz (nach `--ssh`)

Du bist root im gemieteten Container. Wichtigste Stellen:

```bash
# Modell-Download-Fortschritt (falls noch laed)
ls -lh /workspace/hf_cache/hub/models--*/blobs/
stat -c%s /workspace/hf_cache/hub/models--*/blobs/*.downloadInProgress 2>/dev/null
# (kein .downloadInProgress mehr = Download fertig)

# llama-server-Log: Model-Load, Requests, Errors
tail -f /var/log/llama-server.log

# Was laeuft, was lauscht?
pgrep -af 'llama-server|code-server|caddy' | grep -v grep
cat /proc/net/tcp /proc/net/tcp6 | grep ' 0A '   # Listener (ss/netstat fehlen in schlanken Images)

# GPU-Auslastung
nvidia-smi
```

Bei `--with-codeserver`-Boxen (Zusatz-Komponenten):

```bash
# Caddy: TLS / ACME / Routing
tail -f /var/log/caddy.log
cat /etc/caddy/Caddyfile          # gerenderte Config

# code-server: IDE-Startup / Errors
tail -f /var/log/code-server.log

# onstart.sh: wie weit ist Boot gekommen?
cat /var/log/onstart.log
```

Manueller `llama-server`-Restart mit anderen Flags (selten):
siehe [LESSONS.md § Manual llama-server restart](LESSONS.md#manual-llama-server-restart) — env (`LD_LIBRARY_PATH=/app`, `HF_HOME=/workspace/hf_cache`) und `setsid nohup` sind kritisch.

Der Launch druckt Endpoint + API-Key und schreibt `~/.config/opencode/opencode.json` + `~/.local/share/opencode/auth.json`. `opencode` starten — fertig.

## Features

- Ein-Schuss-Launch: Suche → Erstellen → Warten → Config → fertig
- Drei Betriebs-Modi:
  - default — 4 Slots × 16 k Context (kleines Team)
  - `--workshop-mode` — 8 Slots × 16 k Context (5–10 gleichzeitige User)
  - `--solo-mode` — 2 Slots × 64 k Context (single User, lange Sessions)
- **Echter Bandbreiten-Smoketest** — misst die echte HF→Host-Geschwindigkeit vor der langen Wartezeit, probiert automatisch den naechsten Offer wenn zu langsam
- Auto-generierter 32-Char Bearer-Token (Override oder explizit deaktivierbar)
- Kosten-Guards: `--max-price`, `--max-inet-cost`, `--min-inet-down`, `--min-reliability`
- Vast.ai-Template-Support (`--template-hash`) fuer wiederverwendbare Setups
- Diagnose-Modus (`--debug`) zeigt genau welcher Filter die Treffer auf 0 zieht
- Convenience: `--list`, `--destroy ID`, `--ssh ID` (interaktive Shell auf eine laufende Instanz), `--dry-run`

## Voraussetzungen

- Python ≥ 3.10
- `vastai` SDK + `requests` (`pip install vastai requests`)
- Vast.ai-Account mit API-Key in `VAST_API_KEY` (aktueller GPU-Marketplace; siehe [Roadmap](#roadmap))
- Fuer den Smoketest: `ssh-keygen` + `ssh` (Linux/macOS Standard)

## Modi

| Flag | parallel | ctx (gesamt) | pro Slot | Use-Case |
|---|---|---|---|---|
| (default) | 4 | 65 536 | 16 384 | kleines Team, mittellange Chats |
| `--workshop-mode` | 8 | 131 072 | 16 384 | 5–10 User gleichzeitig, Demo |
| `--solo-mode` | 2 | 131 072 | 65 536 | Solo, langer Context |

Override mit `--parallel N --num-ctx M`. Auf 32 GB-Karten ist mit q8 KV-Cache praktisch ~256 k Total-Context die Decke.

## Bandbreiten-Smoketest

Gemietete GPU-Hosts werben mit einer Netzwerk-Geschwindigkeit beim Offer, aber die *echte* HuggingFace-Download-Bandbreite schwankt stark je nach Route, Tageszeit und HF-Mirror-Status. Wir haben 88 Mbps real auf einem "200 Mbps advertised" Host gemessen — mehr als 50 % Shortfall ist haeufig.

`--min-real-mbps N` aktiviert die echte Messung:

1. Container startet, `llama-server` faengt an die GGUF zu laden
2. Launcher SSHt rein (ed25519-Key wird einmalig nach `~/.ssh/gpu_summon_smoketest` generiert + per Marketplace-API attached)
3. Polled bis `.downloadInProgress` Cache-Datei auftaucht (max 90 s)
4. Sample File-Wachstum 15 s lang → echte Mbps
5. Wenn unter Threshold: destroy und naechster Offer aus den Top-5
6. Nach 5 Misses: Exit mit Tipp

Kosten: ~$0.013 pro fehlgeschlagenem Kandidaten. Worst Case (alle 5 fallen durch): ~$0.06 + 8 Min Wartezeit.

```bash
# Strikt — nur Hosts die echte ≥200 Mbps liefern
python summon.py --solo-mode --min-inet-down 500 --min-real-mbps 200

# Locker — 100 Mbps reichen
python summon.py --solo-mode --min-real-mbps 100
```

`--min-inet-down` filtert nach *beworbener* Bandbreite (kostenlos, kein Container); `--min-real-mbps` validiert die *echte* Bandbreite (Container wird erstellt, ggf. zerstoert + retry).

## Authentifizierung

Defaultmaessig generiert der Launcher einen 32-char Bearer-Token und gibt ihn `llama-server` als `--api-key`. Sonst ist dein Endpoint ein public erreichbares Inferenz-Target auf fremder GPU-Rechnung.

```bash
# Default: auto-generieren
python summon.py --solo-mode
# [auth]   Kein --llm-api-key angegeben - generiert: ZeFiOs...

# Festen Key wiederverwenden
GPU_SUMMON_LLM_API_KEY=meinkey python summon.py --solo-mode

# Bewusst offen (nur in vertrauenswuerdigen Netzen!)
python summon.py --solo-mode --llm-api-key ''
```

Die opencode-Auth-Datei (`~/.local/share/opencode/auth.json`) wird mit dem gleichen Token geschrieben.

Hinweis: `/v1/models` ist *nicht* von llama.cpps Auth geschuetzt — nur die Inferenz-Endpoints (`/v1/chat/completions`, `/completion`). Das ist upstream so, kein Bug hier.

## opencode Integration

Zwei Dateien werden via `--write-config` (default on) geschrieben:

- `~/.config/opencode/opencode.json` — Provider-Definition (baseURL, model id, Tools an)
- `~/.local/share/opencode/auth.json` — Bearer-Token

Nach dem Launch einfach `opencode` starten. Provider-ID: `vast-llamacpp`, Model-ID wird aus `/v1/models` automatisch geholt.

## Kosten

Zahlen von vast.ai (April 2026); andere Anbieter haben aehnliche Groessenordnung.

| Was | Typisch |
|---|---|
| Stunde | $0.30–0.40/h auf RTX 5090 |
| Modell-Download (~21 GB UD-Q5_K_XL) | $0.02–0.10 einmalig |
| 1 h interaktives Coding | ~$0.40 |
| 8 h Workshop-Session | ~$3 |
| Smoketest-Probe (pro fehlgeschlagenem Kandidaten) | ~$0.013 |

`--max-price 0.45 --max-inet-cost 0.01` setzt das als harte Filter durch — Over-Budget-Offers werden gar nicht erst geprueft.

## CLI Reference

```
--backend {llamacpp,ollama}    Inferenz-Engine (default: llamacpp)
--model SPEC                   z.B. unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL
--min-vram N                   GPU-VRAM min in GB (default: 32)
--max-price F                  $/h max (default: 0.50)
--min-reliability F            min Reliability (default: 0.95)
--min-inet-down N              min beworbene Mbps (default: 200)
--max-inet-cost F              max Bandbreitenkosten in $/GB (default: 0.01)
--min-real-mbps F              echter Bandbreiten-Smoketest; 0 = aus (default: 0)
--cuda-min F                   min CUDA Version (default: 12.0; 13.2 immer blockiert)
--region CODE                  Geo-Filter, z.B. EU, US (default: any)
--disk N                       Disk in GB (default: 80)

--num-ctx N                    Gesamt-Context-Window (default: 65 536)
--parallel N                   parallele Slots (default: 4)
--workshop-mode                Shortcut: parallel=8, ctx=131072
--solo-mode                    Shortcut: parallel=2, ctx=131072

--llm-api-key TOKEN            Bearer-Token; default auto-generiert; '' = ohne Auth
--template-hash HASH           Vast.ai Template benutzen statt self-rendered onstart

--label NAME                   Instance-Label (default: opencode-llm)
--timeout N                    Container-Up Timeout (default: 600 s)
--model-timeout N              Modell-Ready Timeout (default: 1800 s)
--write-config                 opencode-Config schreiben (default on)
--dry-run                      Plan zeigen, nichts erstellen

--list                         aktive Instanzen listen
--destroy ID                   Instanz zerstoeren
--ssh ID                       SSH in eine laufende Instanz (Key wird auto-attached)
--debug                        Filter-Chain der Offer-Suche analysieren
```

Environment Variables:

- `VAST_API_KEY` — Vast.ai API-Key (required)
- `GPU_SUMMON_LLM_API_KEY` — wiederverwendbarer Bearer-Token fuer den Endpoint
- `GPU_SUMMON_TEMPLATE_HASH` — gepinneter Vast.ai Template-Hash

## Geschriebene Dateien

- `~/.config/opencode/opencode.json` — opencode Provider-Config
- `~/.local/share/opencode/auth.json` — opencode Auth-Tokens
- `~/.ssh/gpu_summon_smoketest{,.pub}` — auto-generiert, nur wenn `--min-real-mbps > 0`

## Vast.ai Templates (Advanced)

Vast.ai-spezifisch. Eine funktionierende Konfiguration kann als Template erfasst und ueber Launches hinweg wiederverwendet werden:

```bash
GPU_SUMMON_TEMPLATE_HASH=<hash> python summon.py --solo-mode
```

Der Launcher nimmt Image + onstart aus dem Template und injected nur Mode-spezifische Env (`LLAMA_PARALLEL`, `LLAMA_CTX`, `LLAMA_MODEL`, `LLAMA_API_KEY`). Wie das Template angelegt wird steht in [LESSONS.md](LESSONS.md#vastai-templates).

## Troubleshooting

**"Keine passenden Offers gefunden"** — Filter lockern: `--max-price 0.50 --min-reliability 0.92`, `--region` weglassen. Mit `--debug` siehst du genau welcher Filter die Treffer killt.

**Endpoint haengt 10+ Min auf HTTP 000** — Modell laedt noch. llama.cpps `-hf` ist still; per SSH `stat -c%s` auf das `.downloadInProgress`-File checken. `--min-real-mbps` benutzen um langsame Hosts frueh rauszuwerfen.

**Container crasht mit `libllama-common.so.0` nicht gefunden** — passiert beim manuellen Restart von llama-server im Container. `LD_LIBRARY_PATH=/app` und `HF_HOME=/workspace/hf_cache` setzen.

**`Bind for :::PORT failed: port is already allocated`** — kaputter Host. Destroyen + Retry; der Pool ist gross genug dass ein schlechter Host kein Problem ist.

Tieferes Diagnose-Material: siehe [LESSONS.md](LESSONS.md).

## code-server-Mode (`--with-codeserver`)

Auf derselben gemieteten Maschine laeuft **LLM und browserbasiertes VS Code** ([code-server](https://github.com/coder/code-server)) zusammen, vor Caddy mit Wildcard-TLS via duckdns + Let's Encrypt. Eine URL, keine lokale Installation, voll TLS, mit code-server's eingebauter Login-Page.

```bash
# Einmaliges Setup auf https://www.duckdns.org/ (kostenlos, ohne Kreditkarte):
#   1. Einloggen (GitHub/Google/Twitter/Reddit OAuth)
#   2. Freie Subdomain waehlen — das Formular zeigt sofort ob noch frei
#   3. Account-Token oben auf der Seite kopieren
# `mybox` unten ist nur ein Beispiel — ueberall *deinen* gewaehlten Namen einsetzen.
export DUCKDNS_TOKEN=...
export VAST_API_KEY=...

# llama-server + code-server + Caddy auf einer gemieteten GPU
python summon.py --with-codeserver --code-domain mybox --solo-mode
# → Browser-IDE:   https://mybox.duckdns.org      (Login: <pw wird gedruckt>)
# → LLM-Endpoint:  https://llm.mybox.duckdns.org/v1   (Bearer: <key wird gedruckt>)

# Aufraeumen (zerstoert Instanz UND raeumt den duckdns-A-Record auf)
python summon.py --destroy <id> --duckdns-token $DUCKDNS_TOKEN
```

Passen Token und Subdomain nicht zusammen (Tippfehler oder Subdomain nie registriert), liefert das erste DNS-Update `KO` zurueck und Caddys Cert-Anforderung scheitert — also erst die Subdomain bei duckdns anlegen, dann launchen.

**Was auf der Box laeuft:**

```
┌─ vast.ai container (kein Docker, alles nativ) ┐
│  llama-server (Host-Prozess, GPU)             │
│  code-server (Host-Prozess, Passwort-Auth)    │
│  Caddy :443 → TLS, Routing                    │
└───────────────────────────────────────────────┘
```

Ursprünglich war Docker-in-Docker geplant fuer Per-Projekt-Workspace-Isolation, aber vast.ai erlaubt `--privileged` auf shared Hosts nicht — `dockerd` kann iptables nicht setup'pen und stirbt beim Boot. Details: [LESSONS.md § code-server Mode](LESSONS.md#code-server-mode-with-codeserver).

**Zusaetzliche Voraussetzungen ggue. Standard-Mode:**

- duckdns.org Subdomain + Token (kostenlos)
- Custom-Image `ghcr.io/dg1001/gpu-summon-codeserver:latest` (einmalig via `codeserver/build-and-push.sh` oder GitHub Actions Workflow gebaut)
- `--disk` wird automatisch auf 60 GB hochgesetzt fuer Image + Modell-Cache + IDE-State

## Roadmap

- [ ] **TensorDock-Backend (naechstes Vorhaben)**. Deren KVM-VM-Modell gibt eine echte oeffentliche IP mit freiem Port 443 und erlaubt `--privileged`-Container — das loest zwei Pain-Points die wir auf vast.ai hatten: (1) den Random-Host-Port-Suffix den der VS-Code-Ports-Tab strippt, und (2) Docker-in-Docker, womit das urspruenglich geplante full-xaresaicoder Per-Projekt-Workspace-Modell als `--with-xares`-Modus neben `--with-codeserver` wieder moeglich wird.
- [ ] Provider-Abstraktions-Layer in `summon.py` (vast/TensorDock/RunPod/Lambda als pluggable Backends mit gemeinsamem CLI)
- [ ] Optional vLLM / SGLang als Backends neben llama.cpp
- [ ] Native CLI-Entry-Point (`gpu-summon` statt `python summon.py`)
- [x] Optionaler Reverse-Proxy mit TLS-Termination (`--with-codeserver` bringt Caddy + duckdns Wildcard-Cert mit)

## Contributing

PRs willkommen. Bitte:

- Mit `--dry-run` testen bevor echte Launches
- Nicht-offensichtliche Erkenntnisse in [LESSONS.md](LESSONS.md) dokumentieren
- Aenderungen klein halten — das ist ein Single-File-Launcher, kein Framework

## Lizenz

MIT — siehe [LICENSE](LICENSE).
