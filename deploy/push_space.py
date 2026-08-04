"""Create (once) and update a private Hugging Face Space running the design studio.

    hf auth login                       # device flow, once — token stays in ~/.cache/huggingface
    python deploy/push_space.py --repo <user-or-org>/biocast-studio

Why a script and not `hf upload`: two details need doing every time and are easy to get
wrong by hand.

1. The Space card is `deploy/space_README.md`, uploaded AS `README.md`. The repository's
   own README has no Space frontmatter, and prepending YAML to it would leave a raw
   `---` block on the GitHub page. So the repo README is excluded and the card is
   uploaded in its place.
2. `biocast/.venv` is a 13 MB virtualenv sitting inside the package directory. Nothing
   ignores it for you here — `.gitignore` does not apply to Hub uploads — and pushing it
   would both bloat the Space and shadow the real dependencies.

Defaults to private, because the source and the literature parameter files travel with
the Space and the project is unpublished. Pass --public deliberately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

#: Kept out of the Space. The runtime needs the package and data/*.json and nothing
#: else; stl/ and docs/ are 10 MB of regenerable meshes and figures.
IGNORE = [
    ".git*", ".git/*", ".github/*", ".claude/*",
    "**/.venv/*", "**/__pycache__/*", "**/*.pyc",
    "stl/*", "docs/*", "examples/*", "deploy/*", "out/*",
    "README.md",                      # replaced by deploy/space_README.md
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True,
                    help="Space id, e.g. yuvalkg/biocast-studio")
    ap.add_argument("--public", action="store_true",
                    help="create the Space publicly — this publishes the source")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    args = ap.parse_args()

    if args.dry_run:                       # must work before `hf auth login`
        files = [p for p in sorted(ROOT.rglob("*")) if p.is_file()]
        keep = [p.relative_to(ROOT) for p in files
                if not any(part in {".git", ".venv", "__pycache__", "stl", "docs",
                                    "examples", "deploy", "out", ".claude", ".github"}
                           for part in p.relative_to(ROOT).parts)
                and p.name != "README.md"]
        total = sum((ROOT / p).stat().st_size for p in keep)
        for p in keep:
            print(f"  {p}")
        print(f"{len(keep)} files, {total/1e6:.1f} MB (+ deploy/space_README.md "
              "as README.md)")
        return

    api = HfApi()
    print(f"authenticated as {api.whoami()['name']}")

    url = api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="docker",
                          private=not args.public, exist_ok=True)
    print(f"space: {url}")

    api.upload_folder(folder_path=str(ROOT), repo_id=args.repo, repo_type="space",
                      ignore_patterns=IGNORE,
                      commit_message="deploy design studio")
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "space_README.md"),
                    path_in_repo="README.md", repo_id=args.repo, repo_type="space",
                    commit_message="space card")
    print(f"pushed. build logs: https://huggingface.co/spaces/{args.repo}?logs=build")


if __name__ == "__main__":
    main()
