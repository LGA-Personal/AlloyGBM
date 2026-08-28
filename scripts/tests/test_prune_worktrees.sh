#!/usr/bin/env bash
# Tests for scripts/prune_worktrees.sh — runs against throwaway temp repos.
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/prune_worktrees.sh"
fails=0
check() { # check <description> <condition-exit-code>
  if [[ "$2" -eq 0 ]]; then echo "ok   - $1"; else echo "FAIL - $1"; fails=$((fails+1)); fi
}

# Build a fresh repo with one commit on main, isolated from the user's git config.
new_repo() {
  local d; d="$(mktemp -d)"
  git -C "$d" init -q -b main
  git -C "$d" config user.email t@t.t
  git -C "$d" config user.name t
  git -C "$d" config commit.gpgsign false
  echo a > "$d/a"; git -C "$d" add a; git -C "$d" commit -qm init
  echo "$d"
}

# --- merged worktree is removed ---
R="$(new_repo)"
git -C "$R" worktree add -q "$R/.worktrees/feat" -b feat >/dev/null
echo b > "$R/.worktrees/feat/b"; git -C "$R/.worktrees/feat" add b; git -C "$R/.worktrees/feat" commit -qm b
git -C "$R" merge -q --no-edit feat
( cd "$R" && "$SCRIPT" >/dev/null )
[[ ! -d "$R/.worktrees/feat" ]]; check "merged worktree removed" $?

# --- unmerged worktree is kept ---
R="$(new_repo)"
git -C "$R" worktree add -q "$R/.worktrees/keep" -b keep >/dev/null
echo c > "$R/.worktrees/keep/c"; git -C "$R/.worktrees/keep" add c; git -C "$R/.worktrees/keep" commit -qm c
( cd "$R" && "$SCRIPT" >/dev/null )
[[ -d "$R/.worktrees/keep" ]]; check "unmerged worktree kept" $?

# --- dirty merged worktree is skipped ---
R="$(new_repo)"
git -C "$R" worktree add -q "$R/.worktrees/dirty" -b dirty >/dev/null
git -C "$R" merge -q --no-edit dirty
echo uncommitted > "$R/.worktrees/dirty/scratch"
( cd "$R" && "$SCRIPT" >/dev/null )
[[ -d "$R/.worktrees/dirty" ]]; check "dirty merged worktree skipped" $?

# --- dry-run changes nothing ---
R="$(new_repo)"
git -C "$R" worktree add -q "$R/.worktrees/dryfeat" -b dryfeat >/dev/null
git -C "$R" merge -q --no-edit dryfeat
( cd "$R" && "$SCRIPT" --dry-run >/dev/null )
[[ -d "$R/.worktrees/dryfeat" ]]; check "dry-run keeps worktree" $?

# --- primary worktree is never removed ---
R="$(new_repo)"
( cd "$R" && "$SCRIPT" >/dev/null )
[[ -d "$R/.git" && -f "$R/a" ]]; check "primary worktree untouched" $?

if [[ "$fails" -gt 0 ]]; then echo "$fails test(s) failed"; exit 1; fi
echo "all prune_worktrees tests passed"
