"""Fail only on NEW ruff findings — the ratchet behind .github/workflows/lint.yaml.

Why not simply `ruff check src tests scripts`: `develop` carries a backlog of pre-existing
findings (mostly E501 on prose inside prompt strings, plus exception-naming and
deliberate blind excepts). Gating the whole tree would fail every
PR for findings its author did not introduce.

Why not "lint the files this PR touches" either — the obvious alternative, and the
one this script replaced: a one-line import reorder in a legacy file makes its
author responsible for that file's entire history. Measured on the very PR that
added this gate: 12 changed files carried 7 findings that predated the change.

So the unit is (file, rule), counted before and after:

  * a pair whose count RISES     -> new finding, fail
  * a pair that APPEARS          -> new finding, fail
  * a pair whose count falls     -> progress, fine
  * everything else unchanged    -> fine

Counting per pair rather than per line makes it immune to line shifts: moving code
without changing it moves every finding's line number but no pair's count. Keying
on the rule as well as the file means a fix in one file cannot mask a regression in
another.

Usage:  python scripts/ruff_ratchet.py <base-ref>
        (run from the repo root, with the base ref fetched)
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys

TARGETS = ("src", "tests", "scripts")


def _findings(ref: str | None) -> collections.Counter:
    """(file, rule) -> count, at `ref` (or the working tree when None).

    Ruff runs against a `git stash`-free checkout of the ref via `git worktree`, so
    the caller's tree is never touched — a CI job that mutated the checkout it was
    about to build would be its own kind of bug.
    """
    if ref is None:
        out = _ruff(".")
    else:
        worktree = ".ruff-ratchet-base"
        subprocess.run(["git", "worktree", "remove", "--force", worktree],
                       capture_output=True, check=False)
        subprocess.run(["git", "worktree", "add", "-f", "--detach", worktree, ref],
                       capture_output=True, check=True)
        try:
            out = _ruff(worktree)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", worktree],
                           capture_output=True, check=False)
    counts: collections.Counter = collections.Counter()
    for item in out:
        # `filename` is absolute; make it repo-relative so base and head compare.
        name = item.get("filename", "")
        for target in TARGETS:
            marker = f"/{target}/"
            if marker in name:
                name = target + "/" + name.split(marker, 1)[1]
                break
        counts[(name, item.get("code") or "?")] += 1
    return counts


def _ruff(root: str) -> list[dict]:
    """Ruff's JSON findings for TARGETS under `root`. A non-zero exit is expected
    (findings exist); only a missing binary or a config error is fatal."""
    paths = [f"{root}/{t}" if root != "." else t for t in TARGETS]
    proc = subprocess.run(
        ["ruff", "check", *paths, "--output-format", "json", "--no-cache"],
        capture_output=True, text=True, check=False,
    )
    if not proc.stdout.strip():
        if proc.returncode not in (0, 1):
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"ruff failed to run (exit {proc.returncode})")
        return []
    return json.loads(proc.stdout)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: ruff_ratchet.py <base-ref>")
    base_ref = sys.argv[1]
    base = _findings(base_ref)
    head = _findings(None)

    regressions = {
        pair: (head[pair], base.get(pair, 0))
        for pair in head
        if head[pair] > base.get(pair, 0)
    }
    fixed = sum(max(0, base[p] - head.get(p, 0)) for p in base)

    print(f"ruff findings: base {sum(base.values())} -> head {sum(head.values())}")
    if fixed:
        print(f"  {fixed} finding(s) cleared — thank you")
    if not regressions:
        print("No new findings. Ratchet holds.")
        return 0
    print("\nNEW ruff findings introduced by this change:\n")
    for (path, rule), (now, before) in sorted(regressions.items()):
        print(f"  {path}: {rule}  {before} -> {now}")
    print("\nFix these, or if a finding is a genuine false positive add a scoped")
    print("per-file-ignore in pyproject.toml with a comment saying why.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
