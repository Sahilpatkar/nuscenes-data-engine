# Workspace boundaries — STRICT

## The rule

**Only `/home/mgaur/sahil/` (and everything beneath it) is writable.**
**Everything outside `/home/mgaur/sahil/` is READ-ONLY. No exceptions.**

This applies to every tool, every command, and every session.

## What "read-only outside `/home/mgaur/sahil/`" means

Outside the writable root, you may **read, list, inspect, and analyze** — nothing else.
The following are **forbidden** anywhere outside `/home/mgaur/sahil/`:

- Creating, writing, editing, appending, or truncating files
- Deleting, moving, renaming, or copying *into* those locations
- `chmod`, `chown`, `mkdir`, `rmdir`, `ln`, `touch`, or any metadata change
- Redirecting output into a path (`>`, `>>`, `tee`), or piping into a writing command
- Any `git` operation that writes outside the repo, or that modifies another repo
- Installing to, or mutating, system/user locations (e.g. `~/.config`, conda/site-packages **outside** this project's `.venv`)

Reading is always fine: `cat`, `ls`, `head`, `find`, `Read`, `Grep`, opening datasets, etc.

## Specifically for this project

- **nuScenes data at `/data/ggare/datasets/nuscenes/` is READ-ONLY.** Never write,
  move, delete, re-encode, or reorganize it. All derived/processed data is written
  **inside** the repo (e.g. `data/processed/`).
- Harness-designated locations are the only allowed writes outside the root, and only
  for their intended purpose: the **scratchpad** directory (throwaway temp files) and
  the Claude Code **memory store**. Never use either as a way around this rule for
  project or dataset files.

## If a task seems to require writing outside the boundary

**Stop and ask the user.** Do not work around it. Propose an in-boundary alternative
(e.g. write outputs under the repo, symlink read-only sources in) and let the user
decide.
