#!/usr/bin/env bash
# Run a repo CLI command on the GPU server at the CURRENT local branch — no manual
# ssh/pull. Pushes the branch, syncs the remote checkout, and runs the command:
#
#   scripts/gpu-run.sh embed --limit-scenes 5          # foreground
#   scripts/gpu-run.sh --bg embed                      # long job: nohup + log file
#   scripts/gpu-run.sh raw "nvidia-smi | head -5"      # arbitrary shell on the node
#   GPU_NODE=trinity-2-3 GPU_DEVICES=3 scripts/gpu-run.sh embed
#
# Env overrides: GPU_NODE (default trinity-2-18 — the head node TRINITY has no GPU
# driver), GPU_REPO, GPU_DEVICES (CUDA_VISIBLE_DEVICES, default 0), GPU_EXTRAS.
# See docs/GPU_SERVER.md.
set -euo pipefail

NODE="${GPU_NODE:-trinity-2-18}"
REPO="${GPU_REPO:-/home/mgaur/sahil/nuscenes_project}"
DEVICES="${GPU_DEVICES:-0}"
EXTRAS="${GPU_EXTRAS:---extra dev --extra data --extra train --extra engine}"

BG=0
if [ "${1:-}" = "--bg" ]; then BG=1; shift; fi
if [ $# -eq 0 ]; then grep '^#' "$0" | head -12; exit 1; fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "!! Uncommitted local changes — the GPU server runs the PUSHED branch." >&2
fi
git push --quiet origin "$BRANCH"
echo ">> $NODE : $REPO @ $BRANCH (CUDA_VISIBLE_DEVICES=$DEVICES)"

if [ "$1" = "raw" ]; then
  shift
  REMOTE_CMD="$*"
else
  # %q-quote each arg so multi-word queries survive the remote shell. `env` (not a
  # bare VAR=val prefix) so the command also works under `nohup` in --bg mode.
  REMOTE_CMD="env CUDA_VISIBLE_DEVICES=$DEVICES uv run nuscenes-data-engine$(printf ' %q' "$@")"
fi

SETUP="cd $REPO && git fetch origin --quiet && git checkout --quiet $BRANCH \
  && git pull --quiet origin $BRANCH && uv sync $EXTRAS --quiet"

if [ "$BG" = 1 ]; then
  LOG="gpu-run-$(date +%Y%m%d-%H%M%S).log"
  # -f backgrounds ssh itself after nohup detaches, so the job survives disconnects.
  ssh -f -o BatchMode=yes "$NODE" "$SETUP && nohup $REMOTE_CMD > $LOG 2>&1 &"
  echo ">> Detached. Follow with:  ssh $NODE tail -f $REPO/$LOG"
else
  TTY_FLAG=""
  [ -t 1 ] && TTY_FLAG="-t"  # tty passthrough (colors, Ctrl-C) only when interactive
  ssh $TTY_FLAG -o BatchMode=yes "$NODE" "$SETUP && $REMOTE_CMD"
fi
