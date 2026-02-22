# Architecture Planning System

This directory uses a nested planning structure so implementation can proceed from broad strategy to concrete tasks without losing context.

## Purpose
- Keep long-term roadmap, phase plans, and sub-plans in one predictable hierarchy.
- Let new agents quickly understand current stage, scope, and next actions.
- Make updates incremental: refine lower levels without rewriting higher-level intent.

## Directory Structure
- `docs/architecture/gpu_financial_gbm_roadmap.md`
  - Global roadmap across all major phases (CPU -> CUDA -> Metal/MLX).
- `docs/architecture/vX.Y/plan.md`
  - Plan for that version level (example: `v1.0` phase plan).
- `docs/architecture/vX.Y/vA.B/plan.md`
  - Child plan that decomposes the parent plan (example: `v1.0/v0.1`).
- `docs/architecture/vX.Y/vA.B/vC.D/`
  - Next decomposition level for implementation slices.

Current active example:
- `v1.0` contains the full phase plan (`0.1.0 -> 1.0.0`).
- `v1.0/v0.1` contains the focused `0.1` implementation plan.
- `v1.0/v0.1/v0.0.1` is reserved for the next sub-plan.

## Planning Rules
1. Each level must have exactly one `plan.md` (or temporary placeholder if not planned yet).
2. Parent plans define goals, boundaries, and release gates.
3. Child plans define implementation sequence, acceptance criteria, and immediate execution scope.
4. Lower-level plans must not contradict parent constraints; if they do, update parent first.
5. Do not skip levels when adding detail. Decompose step-by-step.

## How Agents Should Use This
1. Read `gpu_financial_gbm_roadmap.md` for global direction.
2. Read the highest relevant `plan.md` for current phase constraints.
3. Read the deepest existing child `plan.md` for immediate execution details.
4. Implement against the deepest plan only, while honoring parent constraints.
5. When new detail is needed, create the next child-level plan before implementation.

## Update Workflow
1. Adjust roadmap only for strategic changes.
2. Adjust version-level plan for scope and milestone changes.
3. Add or update child plans for sprint-level/task-level execution.
4. Record assumptions explicitly in the plan where decisions are made.

## Definition of Done for a Plan Level
A plan level is ready when it includes:
- Objective and scope
- In-scope and out-of-scope items
- Interfaces/contracts affected
- Execution sequence
- Test/acceptance criteria
- Risks and mitigations

## Notes
- This system is documentation-first and implementation-guiding.
- It is intended to be attached/shared with fresh agents so they can align quickly.
