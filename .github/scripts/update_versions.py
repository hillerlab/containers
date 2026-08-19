#!/usr/bin/env python3
"""Poll tracked upstream GitHub repos and open a VERSION-bump PR per image.

Triggered on a schedule by .github/workflows/update-versions.yml; run with
--dry-run to preview decisions without touching the repo or opening PRs.

Reads the curated list in .github/upstreams.json:
    { "images": { "<image_dir>": { "repo": "owner/name" } } }

For every entry it compares the image's images/<name>/VERSION against the
upstream's latest release tag (leading "v" ignored). When the upstream is
newer it bumps VERSION on a fresh update/<image>-<target> branch and opens one
PR for that image. It never downgrades, never re-opens for an existing branch,
and skips repos that have no releases.

Requires the `gh` CLI (preinstalled on ubuntu-latest) with GH_TOKEN set; the
runner must have `contents: write` and `pull-requests: write` permissions.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".github" / "upstreams.json"
VERSION_RE = re.compile(r"^[vV]")
NUM_RE = re.compile(r"[^0-9]+")


def log(message: str) -> None:
    print(message, flush=True)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command in the repo root, inheriting the runner's env (GH_TOKEN)."""
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=check)


def gh_output(*args: str) -> str | None:
    """`gh api ...` stdout (stripped), or None when it fails."""
    try:
        return run("gh", "api", *args).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def latest_release_tag(repo: str) -> str | None:
    """Newest stable release tag, falling back to the most recent tag."""
    tag = gh_output(f"repos/{repo}/releases/latest", "--jq", ".tag_name")
    if tag:
        return tag
    return gh_output(f"repos/{repo}/tags", "--jq", ".[0].name")


def version_key(version: str) -> tuple:
    """Numeric-ish compare so '1.10' > '1.9' and '1.03' == '1.3'."""
    return tuple(int(part) for part in NUM_RE.split(version) if part)


def bump_image(image: str, repo: str, dry_run: bool) -> None:
    version_file = ROOT / "images" / image / "VERSION"
    if not version_file.is_file():
        log(f"[error] images/{image}/VERSION missing")
        raise SystemExit(1)

    current = version_file.read_text().strip()
    tag = latest_release_tag(repo)
    if tag is None:
        log(f"[skip] {image}: {repo} has no releases or tags to watch")
        return

    target = VERSION_RE.sub("", tag).strip()
    if target == current:
        log(f"[skip] {image}: already in sync at {current}")
        return
    if not target or not re.fullmatch(r"[A-Za-z0-9_.+-]+", target):
        log(f"[skip] {image}: cannot map release tag {tag!r} to a VERSION")
        return
    if version_key(current) > version_key(target):
        log(f"[skip] {image}: local VERSION {current} is already ahead of {tag}")
        return

    branch = f"update/{image}-{target}"
    existing = run("git", "ls-remote", "--heads", "origin", branch, check=False)
    if existing.stdout.split():
        log(f"[skip] {image}: branch {branch} already exists (PR open or stale)")
        return

    log(f"[bump] {image}: {current} -> {target} (from {repo} release {tag})")
    if dry_run:
        return

    version_file.write_text(target + "\n")
    run("git", "checkout", "-b", branch)
    run("git", "add", f"images/{image}/VERSION")
    run("git", "commit", "-m", f"feat: bump {image} {current} -> {target}")
    run("git", "push", "-u", "origin", branch)

    needs_checksum = (ROOT / "images" / image / "SHA256SUMS").is_file()
    body = (
        f"This image tracks [{repo}](https://github.com/{repo}).\n\n"
        f"Upstream released `{tag}`; VERSION here is `{current}`. "
        f"This PR bumps `images/{image}/VERSION` to `{target}`.\n\n"
        f"- [ ] `docker build` succeeds and the binary reports `{target}`\n"
    )
    if needs_checksum:
        body += (
            f"- [ ] Refresh `images/{image}/SHA256SUMS` for the new release "
            "(the Dockerfile verifies its downloads against this file)\n"
        )
    body += (
        "- [ ] Merge; the publish workflow builds changed images on the next "
        "`release`/tag of this repo\n"
    )
    run(
        "gh",
        "pr",
        "create",
        "--title",
        f"Bump {image} {current} -> {target}",
        "--body",
        body,
    )


def main() -> None:
    dry_run = "--dry-run" in sys.argv[1:]
    config = json.loads(CONFIG.read_text())
    for image, entry in config["images"].items():
        bump_image(image, entry["repo"], dry_run)
    log("update-versions: done")


if __name__ == "__main__":
    main()
