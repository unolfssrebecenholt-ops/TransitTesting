#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parents[index] for index in range(1, min(5, len(script_path.parents)))]
    for candidate in candidates:
        if (candidate / "app.py").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise SystemExit("Could not locate repo root containing app.py and pyproject.toml.")


def main() -> int:
    repo_root = find_repo_root()
    cmd = ["uv", "run", "--project", str(repo_root), "app.py", *sys.argv[1:]]
    env = os.environ.copy()
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
