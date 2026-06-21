# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs) — short, dated documents capturing a single significant decision, the alternatives considered, and the reasoning, at the point the decision was made.

No ADRs have been recorded yet; the architecture as currently designed is captured directly in [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) and the frozen specifications in [`docs/specs/`](../specs/README.md).

## When to add one

Add an ADR here when a decision is significant enough that a future engineer would otherwise have to re-derive the reasoning from the code — for example, a change to the offline/online boundary, the artifact contract, the scoring formula's shape, or a layer's import rules. Routine implementation choices that don't affect the architecture described in [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) do not need one.

## Suggested format

```text
NNNN-short-title.md

# NNNN. Title
Date: YYYY-MM-DD
Status: proposed | accepted | superseded by NNNN

## Context
What forces are at play; what problem is being solved.

## Decision
What was decided.

## Consequences
What becomes easier or harder as a result.
```
