"""Lightweight reproducibility helper.

This script provides convenience actions to create a virtualenv and to print or run the canonical
repro steps described in REPRODUCE.md. It is intentionally conservative: by default it only prints
the planned commands (use --apply to execute environment creation and package install).

Usage examples:
  python3 scripts/reproduce.py --dry-run
  python3 scripts/reproduce.py --setup-env --install

This script should be safe to run locally; it will not overwrite project files.
"""
import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))


def run(cmd, check=True):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check)


def setup_env(venv_path='.venv', apply_changes=False):
    venv_dir = os.path.join(ROOT, venv_path)
    if apply_changes:
        print(f"Creating virtualenv at {venv_dir}")
        run(f"python3 -m venv {venv_dir}")
        pip = os.path.join(venv_dir, 'bin', 'pip')
        run(f"{pip} install --upgrade pip")
        run(f"{pip} install -r {os.path.join(ROOT, 'requirements.txt')}")
    else:
        print("Dry-run: would create virtualenv and install requirements:")
        print(f"python3 -m venv {venv_dir}")
        print(f"{venv_dir}/bin/pip install -r requirements.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup-env', action='store_true')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--venv-path', default='.venv')
    args = parser.parse_args()

    apply_changes = args.install and not args.dry_run

    if args.setup_env:
        setup_env(venv_path=args.venv_path, apply_changes=apply_changes)
    else:
        print("No action requested. See --help. Default is --dry-run to inspect steps.")


if __name__ == '__main__':
    main()
