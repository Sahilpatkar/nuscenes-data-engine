# GPU server usage

How to run compute (ingest, train, evaluate, manifest, embed) on the GPU server
**directly from this repo** — no manual ssh + git pull. Everything here assumes the
two-machine topology from the README: the server computes and writes plain files; this
machine (the infra machine) owns Docker, MLflow UI, serving, and monitoring.

## The machines

| Host (ssh alias) | Hardware | Use for |
|---|---|---|
| `TRINITY` (head node) | **no GPU driver** | CPU/NFS-bound work: `ingest`, `validate`, `manifest` |
| `trinity-2-18` | 4× RTX 3080 Ti (12 GB) | default GPU node: `embed`, `train`, `evaluate` |
| `trinity-2-3` | 4× RTX 3090 (24 GB) | bigger-batch training when 2-18 is busy |
| `trinity-0-3` | 4× RTX 6000 Ada (48 GB) | largest jobs (check occupancy first — often shared) |

All nodes share the same home filesystem, so the repo checkout at
`/home/mgaur/sahil/nuscenes_project` and the dataset at
`/data/ggare/datasets/nuscenes` (read-only) are visible everywhere. The compute nodes
are reached via `ProxyJump` through the head node — the ssh aliases in
`~/.ssh/config` handle this:

```
Host TRINITY
  HostName trinity.vision.cs.cmu.edu
  User <you>
Host trinity-2-18            # (and the other compute nodes)
  ProxyJump TRINITY
  User <you>
```

**Etiquette:** check `nvidia-smi` before launching, pin one GPU with
`GPU_DEVICES=<n>` (the runner sets `CUDA_VISIBLE_DEVICES` — default `0`), and prefer
an idle node. All library caches (Hugging Face, Ultralytics) are redirected into the
repo's `.cache/` — nothing writes to `$HOME` outside the project workspace.

## The remote runner — `scripts/gpu-run.sh`

One command from your machine: pushes your **current branch**, fast-forwards the
server checkout to it, `uv sync`s, and runs the CLI there.

```bash
scripts/gpu-run.sh embed --limit-scenes 5     # foreground, streamed to your terminal
scripts/gpu-run.sh --bg embed                 # long job: nohup + gpu-run-<ts>.log, survives disconnects
scripts/gpu-run.sh raw "nvidia-smi | head"    # arbitrary shell on the node
scripts/gpu-run.sh raw "uv run pytest -m engine_smoke -rA"
```

Environment overrides (prefix any invocation):

| Variable | Default | Meaning |
|---|---|---|
| `GPU_NODE` | `trinity-2-18` | which host runs the command |
| `GPU_DEVICES` | `0` | `CUDA_VISIBLE_DEVICES` on the node |
| `GPU_REPO` | `/home/mgaur/sahil/nuscenes_project` | server checkout path |
| `GPU_EXTRAS` | dev+data+train+engine | extras for the remote `uv sync` |

Make shortcuts: `make gpu-embed`, `make gpu-train` (both `--bg`),
`make gpu-manifest` (runs on the head node — CPU/NFS work), and
`make gpu-run CMD="evaluate --imgsz 960"`.

For a `--bg` job, follow progress with the command the runner prints:
`ssh trinity-2-18 tail -f <repo>/gpu-run-<timestamp>.log`.

## Getting outputs back — `make sync-down`

The server only writes plain files; this machine pulls them:

```bash
make sync-down   # data/processed (parquet, excl. the yolo/ symlink tree),
                 # mlruns/ (MLflow store + registry), data/lancedb/ (vector store)
```

Uses `rsync --partial --timeout=120` so a dropped connection fails loudly and resumes
cleanly on retry. Sync only while no job is writing (the embed job compacts its Lance
fragments at the end for exactly this reason). Note: macOS ships openrsync — fancy
GNU-rsync flags like `--info=progress2` don't exist.

## What runs where (cheat sheet)

| Command | Node | Why |
|---|---|---|
| `ingest`, `validate`, `manifest` | `TRINITY` (head) | CPU + NFS metadata scans; no GPU needed |
| `train`, `evaluate` | GPU node | CUDA; multi-GPU via `--device 0,1` |
| `embed` | GPU node | SigLIP over ~205K frames (~85 min on one 3080 Ti); resumable per scene — safe to kill and relaunch |
| `autolabel sample/submit/status/collect` | `TRINITY` (head) | needs the images + outbound HTTPS; network-bound (Claude Batch API) |
| `search`, `query`, `monitor report`, `autolabel eval` | either / this machine | need only the synced store/parquet |
| serving, Streamlit, MLflow UI, Docker | this machine only | the server never runs infra |

## Failure modes seen in practice

- `Found no NVIDIA driver` → you're on the head node; set `GPU_NODE=trinity-2-18`.
- A `--bg` job's log stops growing → check `ssh <node> pgrep -af nuscenes-data-engine`;
  the embed job resumes from its last completed scene on relaunch.
- Hugging Face rate-limit warnings on first model download are harmless; the model
  lands in `<repo>/.cache/huggingface` once and is reused (also by the Docker api
  container via its `.cache` volume mount).
