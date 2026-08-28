#!/usr/bin/env bash
# Prune stale / merged AlloyGBM git worktrees under .worktrees/ and
# .claude/worktrees/. Safe by default: only removes a worktree whose branch is
# merged into the default branch, gone, or detached, AND which has no
# uncommitted changes. Never touches the primary working tree or worktrees
# outside the two managed directories.
#
# Usage:
#   scripts/prune_worktrees.sh            # remove safe stale worktrees
#   scripts/prune_worktrees.sh --dry-run  # print what would be removed only
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

default_branch="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
default_branch="${default_branch:-main}"

# Drop admin entries for worktrees whose directory is already gone.
git worktree prune

removed=0
path=""; branch=""; detached=0

flush() {
  [[ -z "$path" ]] && return 0
  local p="$path" b="$branch" det="$detached"
  path=""; branch=""; detached=0

  # Never touch the primary working tree.
  [[ "$p" == "$repo_root" ]] && return 0
  # Only manage worktrees under the two managed directories.
  case "$p" in
    "$repo_root"/.worktrees/*|"$repo_root"/.claude/worktrees/*) ;;
    *) return 0 ;;
  esac

  local reason=""
  if [[ "$det" == 1 || -z "$b" ]]; then
    reason="detached HEAD"
  elif ! git show-ref --verify --quiet "refs/heads/$b"; then
    reason="branch gone"
  elif git merge-base --is-ancestor "$b" "$default_branch" 2>/dev/null; then
    reason="merged into $default_branch"
  fi
  [[ -z "$reason" ]] && return 0

  if [[ -n "$(git -C "$p" status --porcelain 2>/dev/null || true)" ]]; then
    echo "skip (dirty): $p"
  elif [[ "$DRY_RUN" == 1 ]]; then
    echo "would remove ($reason): $p"
  else
    git worktree remove --force "$p"
    echo "removed ($reason): $p"
    removed=$((removed + 1))
  fi
}

while IFS= read -r line; do
  case "$line" in
    "worktree "*) flush; path="${line#worktree }" ;;
    "branch "*)   branch="${line#branch refs/heads/}" ;;
    "detached")   detached=1 ;;
  esac
done < <(git worktree list --porcelain)
flush

# Remove stray empty leftovers (e.g. macOS .DS_Store) in the managed dirs.
for d in .worktrees .claude/worktrees; do
  [[ -d "$d" ]] || continue
  find "$d" -name .DS_Store -delete 2>/dev/null || true
  rmdir "$d" 2>/dev/null || true
done

echo "worktree prune complete (removed $removed)."
