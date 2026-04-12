# Lab Notes — OCR v2

This file is your required experiment journal. The user will read it.

## Format

For each experiment, add a section with:

```markdown
## YYYY-MM-DD HH:MM — <short experiment name>

**Hypothesis:** Why you think this will work in 1-2 sentences.

**Prediction:** What cross-validation accuracy you expect (a specific
number, not "good" or "better").

**Stop criterion:** What you'd need to see to abandon this approach.

**Setup:** Architecture / hyperparameters / data subset / trim level
in 2-4 lines. Make it reproducible.

**Result:** Actual cross-validation accuracy + any per-digit or
per-phone breakdown that's interesting.

**Right or wrong:** Did your prediction match? By how much? If wrong,
why?

**Learning:** One sentence. What did this experiment teach you that
you didn't know before?

**Next:** What you'd try next, or "abandoning this branch".
```

## Phase 1 — Literature review

(You'll fill this in with one paragraph per PDF in `research/`, and a
ranked GAME_PLAN.md when done.)

## Phase 2 — Data exploration + trim ablation

(You'll fill in observations from `data/train_cells/` + the required
margin-trim ablation results here.)

### Required: trim ablation

Trim percentages to test (per side, before resize to 28×28 or your
chosen input size):

| Trim per side | Cell area kept | KNN/SVM baseline 5-fold CV acc |
|---|---|---|
| 0% (no trim) | 100% | TODO |
| 10% | 64% | TODO |
| 12% | 57.8% | TODO |
| 15% | 49% | TODO |
| 18% | 42.2% | TODO |
| 20% | 36% | TODO |

The user explicitly flagged this as a free hyperparameter that the previous OCR
left at 10% without checking. Find the optimal trim level for THIS
training distribution and use it as your default for the rest of the
project. If best ≠ 10%, the user can apply the same trim to the previous OCR in
the parent pipeline as a free win.

## Phase 3 — Baseline

## Phase 4 — Iteration

## Phase 5 — Final model summary

(Will live in `RESULTS.md` when you're done — but copy the
one-paragraph "why this works and the previous OCR doesn't" answer here too.)

---

## Open questions for the user

If you're stuck on a decision and need the user's input but don't
want to block, write the question here and proceed with your best
guess. The user will read this section and answer in their next
session.

(empty)
