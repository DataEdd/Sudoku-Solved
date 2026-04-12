# OCR v2 — your brief

## Mission

Build an OCR model for Sudoku digit recognition, trained from scratch on Baptiste Wicht's V2 training set, that beats the previous attempt's 61.5% filled-cell accuracy on the V2 test set. The previous attempt was a 102K-parameter custom CNN trained on synthetic data; you are testing the hypothesis that **in-distribution training beats architecture choice** by training on real photographs from the same distribution as the test set.

You have NOT seen the V2 test set. You must not read any test-set file during development. Final evaluation is **gated** — the user runs `scripts/final_eval.py` after you hand them a model. You do not run it yourself, ever.

## Context

The previous OCR (v5.1) was a 102K-parameter custom CNN trained on:
- MNIST handwritten digits 1-9 (label 0 dropped because Sudoku "empty" ≠ digit 0)
- ~4,500 system-font-rendered printed digits from 67 Latin-digit fonts with rotation/noise/blur augmentation
- ~1,800 Chars74K held-out fonts
- ~5,000 synthetic empty-cell variants matching a separate hand-annotated benchmark distribution

When v5.1 is evaluated zero-shot on the V2 test set, it scores:

- **61.5% filled-cell accuracy** (676/1100)
- **99.4% empty-cell accuracy** (very conservative)
- **5/38 perfect images** on the 38 correctly-detected images
- **Failure mode**: 77% of errors are "missed" (predicted empty when GT was filled). Only 6% are confidently wrong. The CNN is conservative on out-of-distribution inputs.
- **Per-phone accuracy** is bimodal: 96% on iPhone 3GS (2009), 37% on Sony Ericsson t660i (2008) at 1600×1200. Failure is concentrated on specific phone/resolution combinations, not uniform.
- **Per-digit accuracy** is non-uniform: digits 8 and 9 are missed most often (~58-59% missed), digit 2 is best (~22% missed). Digits with denser ink suffer more.

The baseline you must beat: **61.5% filled-cell accuracy and/or 5 perfect images on the V2 test set's 40 images**.

## The hypothesis you're testing

1. **In-distribution training matters more than architecture choice.** v5.1's training data was implicitly tuned to a different photo distribution (a separate macfooty-derived benchmark of mostly modern smartphones). The Wicht V2 test set is a mix of 2006-2013 phones, dominated by 2007-era Sony Ericsson at 1600×1200. v5.1 has never seen anything like those cell crops. Training on Wicht's V2 training images (which come from the same phones as the test set) should close most of the 21-point gap.
2. **Cell-extraction trimming is a free hyperparameter.** v5.1's preprocessing trims a 10% margin per side from each cell before resizing to 28×28. This may not be enough — there's likely grid-line residue at cell borders that confuses the classifier. The prep script you're given saved cells with the FULL 1/9 slice (no trim), so the trim is your hyperparameter to tune. **This must be a required ablation in Phase 2 below.**

## Hard constraints

1. **Sandboxed working directory.** You may only read and write files inside the directory this notebook lives in (your cwd). Do not attempt to read any file via absolute paths outside it. There is no parent tree to peek at — the directory was copied here precisely so you cannot.
2. **No test-set access.** You will not find the test set in this directory. There is no `v2_test.desc`, no `image1005.jpg`, no `image91.jpg`, etc. Do not try to obtain them via the network, via pip packages, or via any other channel. If you need to estimate generalization during development, use **k-fold cross-validation on the train cells you have**.
3. **No reuse of v5.1 code.** You don't have access to it from this directory anyway, but for clarity: do not import from `app.ml.recognizer`, `app.core.extraction`, or any other parent-repo module. v2 is a clean reimplementation. You may use any third-party library (PyTorch, scikit-learn, ONNX, etc.).
4. **Self-contained.** Every file your final model needs to load and run lives inside this directory. The user will be able to `cp -r` it back to the main repo and have everything work.
5. **Lab journal required.** Maintain `LAB_NOTES.md` with every experiment. Each entry needs a hypothesis, prediction, result, and one-sentence learning. This is auditable — the user reviews it after you finish.
6. **License.** The training images are CC BY 4.0 from `wichtounet/sudoku_dataset` (the canonical research dataset by Baptiste Wicht / iCoSys, University of Fribourg). Attribute appropriately in any derivative.

## Workflow — follow in order

### Phase 1 — Literature review (~1-2 hours)

Read every PDF in `research/`. For each, write a short summary in `research/summaries/<filename>.md` covering: **method, architecture, training data, reported results, what's reusable, what's obsolete**. The Wicht 2014 paper is the canonical baseline — his V1 dataset is closely related to your training set, and he reported 87.5% perfect-image rate on V1 test using a Deep Belief Network. His Ph.D. thesis reportedly hits 82.5% on V2 test with the same family of methods.

After the literature review, write **`GAME_PLAN.md`** listing 3-5 candidate approaches with tradeoffs. Examples to consider:

- **Modern small CNN** properly hyperparameter-tuned for this distribution (different from v5.1's architecture)
- **Transfer learning from MNIST or SVHN** (pre-train on big data, fine-tune on Wicht train)
- **Classical HOG + SVM** (Wicht's contemporaries — fast to iterate, useful baseline)
- **Wicht's 2014 DBN reimplemented** (historically accurate baseline; PyTorch has RBM modules)
- **Ensemble** combining two fast classifiers with per-digit routing based on cross-validation strengths
- **Something from the literature review you hadn't considered**

For each approach, write: expected accuracy gain, implementation cost (rough hours), main risk, key reuseable component. Rank them by `(expected gain) × (probability of success) / (implementation cost)`. **Commit to one or two** to try first. Do not implement anything yet.

### Phase 2 — Data exploration + the trim ablation (~30-60 minutes)

Only `data/train_cells/` and `data/train_metadata.jsonl`. Look at:

- How many cells total? Filled vs empty distribution?
- Per-digit label distribution across the 9 classes
- Visual sample: render 10 random cells per digit + 10 empty cells. Record observations about cell quality (noise, contrast, font variation, grid-line residue, ambiguous empties).
- Per-phone and per-resolution distribution from `train_metadata.jsonl`
- Any obviously broken cells you should filter out

**The cell-extraction trim ablation is REQUIRED.** Train a quick KNN or SVM baseline on the train cells with each of these margin trims and 5-fold cross-validation:

| Trim per side | Effective cell area kept |
|---|---|
| 0% (no trim) | 100% |
| 10% (matches v5.1) | 64% |
| 12% | 57.8% |
| 15% | 49% |
| 18% | 42.2% |
| 20% | 36% |

Record the cross-validation accuracy at each trim level in `LAB_NOTES.md`. The best trim becomes your default for the rest of the project. Also note whether the best trim varies by digit (e.g. wide digits like 8 might prefer less trim than narrow digits like 1).

Log all data observations + the trim verdict in `LAB_NOTES.md` before touching a deep model.

### Phase 3 — Implement ONE simple baseline (~2-4 hours)

Your simplest plausible candidate from the game plan, not the fanciest. Train it, evaluate via 5-fold CV on the train set (NEVER touch test), log the number. This is your reference point. Any future experiment must beat this baseline by a non-trivial margin (≥ 2 percentage points) or it's not worth pursuing.

### Phase 4 — Iterate the top-ranked approach (~4-12 hours, user-gated)

Run your top-ranked candidate. Each experiment in `LAB_NOTES.md` requires:

1. **Hypothesis** — why you think this will work
2. **Prediction** — what cross-validation accuracy you expect
3. **Stop criterion** — what would you need to see to consider it a fail and abandon

After each experiment, update with the actual result and whether your prediction was right. **Predicting wrong is fine; not predicting at all is the failure mode** — without explicit predictions you can't tell whether you're learning or just stumbling.

When cross-validation accuracy plateaus and your last 2-3 experiments don't improve over the previous best, **stop training and move to Phase 5**. Resist the urge to keep tweaking — the hypothesis is being tested, not the model is being maximized.

### Phase 5 — Final model package (~1 hour)

When the model is ready to ship, write `src/infer.py` exposing:

```python
def recognize_cells(cells: List[np.ndarray]) -> Tuple[List[List[int]], List[List[float]]]:
    """Run v2 OCR on a list of 81 cell images.

    Returns:
      grid: 9x9 list of int (0 = empty, 1-9 = digit)
      conf: 9x9 list of float (max softmax probability over classes 1-9)
    """
```

This is the same interface the parent repo's pipeline uses, so v2 could be a drop-in replacement (the user does that integration after final_eval, not you).

Save the trained weights to `checkpoints/v2.{pth,onnx}` (whichever your model uses). Write `RESULTS.md` summarizing:

1. Cross-validation accuracy on train set (your internal estimate)
2. The chosen architecture in 3-5 sentences (what / why / trained on / parameter count)
3. The trim ablation result from Phase 2
4. What you'd try next if you had another day

Then notify the user. **Do not run `scripts/final_eval.py`** — that's the gate the user uses to compare you against v5.1.

## Success criteria (in priority order)

1. **Cross-validation accuracy on train ≥ 75% filled-cell accuracy.** Below this, your model isn't learning anything in-distribution and there's no point running the test eval.
2. **Beats v5.1's 61.5% filled-cell on V2 test** (the user verifies via `final_eval.py`). This is the headline metric.
3. **Beats v5.1's 5 perfect images** on the 40-image test set (also via `final_eval.py`).
4. **Doesn't catastrophically fail on Sony Ericsson t660i** (10 of the 40 test images, where v5.1 hits 37%). You don't need to ace these, but you should not be 0/10.
5. **Clean failure mode**: either conservative (low hallucinations) or roughly symmetric (wrong ≈ missed). Avoid models with many hallucinations — a single hallucinated digit kills Sudoku solvability.
6. **Explicable**: `LAB_NOTES.md`'s final entry should answer "why does this work and v5.1 doesn't" in one paragraph.

## Anti-success — what NOT to do

- **Do not train on test data, even accidentally.** Cross-validate on train ONLY.
- **Do not import any v5.1 code.** Do not look for it. There's nothing to find.
- **Do not build an 800K-parameter monster.** v5.1 is 102K params at ~40 ms per 81-cell batch. Stay in the same ballpark — under 1M params, under 200 ms per batch. This is a Sudoku OCR, not an OCR for the entire English language.
- **Do not skip the lab journal.** A great model with no journal is rejected. A mediocre model with a complete journal is accepted.
- **Do not skip Phase 2's trim ablation.** This is a required experiment. If it turns out to be a wash, that's a finding worth recording — but you must run it.
- **Do not build an ensemble that includes the user's previous v5.1 model.** You don't have access to it, and that would defeat the entire purpose of testing the in-distribution-training hypothesis.

## Deliverables

When done, the user should be able to run:

```bash
cd ocr_v2
python -m src.infer <image_path>      # standalone usage on one image
cat LAB_NOTES.md                       # complete experiment log
cat GAME_PLAN.md                       # what you tried, ranked
cat RESULTS.md                         # final summary
ls checkpoints/                        # trained weights
```

Plus the user runs `python -m scripts.final_eval` (gated) to produce the official benchmark.

## Tools you have

You're running in Claude Code. You have:

- `Read`, `Edit`, `Write`, `Bash`, `Glob`, `Grep` for file operations
- The full Python ecosystem available via `pip install` if needed (PyTorch, scikit-learn, ONNX, etc. — but install via `pip install` in your venv, not by reading installed packages from outside the sandbox)
- The 4 PDFs in `research/` for the literature review
- ~12,900 pre-extracted labeled cell PNGs in `data/train_cells/`
- Per-image metadata in `data/train_metadata.jsonl`

## When you're stuck

If you find yourself stuck or unsure about a design choice, **write the question in `LAB_NOTES.md` and proceed with your best guess**. Don't loop on the same decision. The user will read the journal and can answer in the next session if needed. Forward progress beats perfect decisions.

Good luck. Match the literature, and where the literature is silent, run the experiment.
