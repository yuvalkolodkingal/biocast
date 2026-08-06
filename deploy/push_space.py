"""Create (once) and update a private Hugging Face Space running the design studio.

    hf auth login                       # device flow, once — token stays in ~/.cache/huggingface
    python deploy/push_space.py --repo <user-or-org>/biocast-studio

Under CI there is no login and no stored token: `.github/workflows/deploy-space.yml`
sets `HF_OIDC_RESOURCE`, and `huggingface_hub` exchanges GitHub's OIDC id token for a
Space-scoped one (Trusted Publishers). This script needs no flag for that — the
exchange happens inside `get_token()` — but it must not assume a personal token, which
is why `whoami` is best-effort and `create_repo` only runs when the Space is absent.

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

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.utils import filter_repo_objects

ROOT = Path(__file__).resolve().parents[1]

#: Kept out of the Space. The runtime needs the package and data/*.json and nothing
#: else; stl/ and docs/ are 10 MB of regenerable meshes and figures.
#: Patterns are fnmatch, applied to the path relative to the repository root, and
#: fnmatch's `*` spans "/". So `.venv/*` covers the whole tree, while a leading `**/`
#: does NOT match at the root — `**/.venv/*` silently misses a root-level `.venv`,
#: which is how a 10k-file virtualenv reached the Space on the first push. Both depths
#: are therefore spelled out, and --dry-run runs these exact patterns.
IGNORE = [
    ".git/*", ".gitignore", ".gitattributes", ".github/*", ".claude/*",
    ".venv/*", "*/.venv/*",
    "__pycache__/*", "*/__pycache__/*", "*.pyc",
    "stl/*", "docs/*", "examples/*", "deploy/*", "out/*",
    "*.egg-info/*", ".ipynb_checkpoints/*",
    "README.md",                      # replaced by deploy/space_README.md
    # Sweep outputs, read only by examples/ and the docs — both already excluded.
    # 5.4 MB of the 6.0 MB that would otherwise ship, on every deploy, into the
    # Docker build context of a container that never opens them.
    "data/design_space.csv", "data/pareto_front.csv",
]


def upload_set() -> list[str]:
    """Exactly what `upload_folder` will send, via the same filter it uses. Kept as one
    function so --dry-run cannot disagree with the real push — the first version of this
    script reimplemented the filtering by hand and reported 32 clean files while the
    upload sent 10,234."""
    rel = (str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file())
    return list(filter_repo_objects(rel, ignore_patterns=IGNORE))


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
        keep = sorted(upload_set())
        total = sum((ROOT / p).stat().st_size for p in keep)
        for p in keep:
            print(f"  {p}")
        print(f"{len(keep)} files, {total/1e6:.1f} MB (+ deploy/space_README.md "
              "as README.md)")
        return

    api = HfApi()

    # BOTH OF THESE HAVE TO TOLERATE A TRUSTED-PUBLISHER TOKEN.
    #
    # Under CI there is no personal token: `HF_OIDC_RESOURCE` makes `get_token()`
    # exchange GitHub's OIDC id token for one scoped to this Space alone, attributed
    # to a synthetic `[OIDC]` user. So `whoami()` is decoration, not a check — it can
    # legitimately fail — and `create_repo` is a namespace-level write the scoped
    # token cannot do. Neither should be able to fail a deploy of a Space that
    # already exists, which under CI it always does.
    try:
        print(f"authenticated as {api.whoami()['name']}")
    except Exception as exc:                   # OIDC token, or none at all
        print(f"authenticated via a scoped token (whoami unavailable: {exc})")

    if api.repo_exists(repo_id=args.repo, repo_type="space"):
        print(f"space: https://huggingface.co/spaces/{args.repo}")
    else:
        url = api.create_repo(repo_id=args.repo, repo_type="space",
                              space_sdk="docker", private=not args.public)
        print(f"space: {url} (created {'public' if args.public else 'private'})")

    # ONE commit, not two, and it deletes.
    #
    # `upload_folder` + a second `upload_file` for the card is two revisions, so the
    # Hub queues two builds seconds apart and there is a window in which the Space
    # runs new code under the old card. It also matters under CI: a wait-for-build
    # gate can watch the first build reach RUNNING and report success while the
    # second is still queued, so a failing deploy goes green.
    #
    # `upload_folder` also only ever ADDS. Rename a module and the old one stays on
    # the Space beside the new one, where Python imports it perfectly happily — a
    # stale driver that keeps serving after the source has moved on, with nothing in
    # the build log to say so. Deletions are computed against the same `upload_set()`
    # that --dry-run prints, so the two cannot disagree.
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(ROOT / p))
           for p in upload_set()]
    ops.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=str(ROOT / "deploy" / "space_README.md")))
    # `.gitattributes` is Hub-generated LFS config that does not exist in this
    # repository; deleting it would strip the Space's LFS tracking.
    #
    # PRUNING IS BEST-EFFORT, because listing needs a permission publishing does not.
    # A Trusted-Publisher token is scoped to write this one Space and cannot READ the
    # file tree of a private one: `list_repo_files` returns 401 while `create_commit`
    # on the same repo with the same token succeeds. Deleting stale files is worth
    # having and is not worth failing a deploy over, so a listing failure downgrades
    # to an add-only push and says so loudly enough to act on.
    keep = {op.path_in_repo for op in ops} | {".gitattributes"}
    try:
        stale = [p for p in api.list_repo_files(args.repo, repo_type="space")
                 if p not in keep]
    except Exception as exc:
        stale, pruned = [], False
        print(f"WARNING: cannot list the Space, so nothing stale can be pruned "
              f"({type(exc).__name__}). This push only ADDS. A file removed from git "
              f"stays on the Space, where Python will still import it. Prune with a "
              f"personal token: python deploy/push_space.py --repo {args.repo}")
    else:
        pruned = True
        ops += [CommitOperationDelete(path_in_repo=p) for p in stale]

    api.create_commit(repo_id=args.repo, repo_type="space", operations=ops,
                      commit_message="deploy design studio")
    print(f"pushed {len(ops) - len(stale)} files"
          + (f", removed {len(stale)}: {', '.join(sorted(stale)[:6])}"
             + ("…" if len(stale) > 6 else "") if stale else
             ("" if pruned else " (add-only)")))
    print(f"build logs: https://huggingface.co/spaces/{args.repo}?logs=build")


if __name__ == "__main__":
    main()
