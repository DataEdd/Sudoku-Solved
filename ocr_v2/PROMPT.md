# OCR v2 — your brief

## Mission

Build an OCR model for Sudoku digit recognition, trained from scratch on the labeled cell dataset in `data/train_cells/`. The training data comes from real newspaper Sudoku photographs taken on a mix of phone cameras spanning roughly 2006-2013. Your goal is to beat a previous synthetic-trained baseline that hits **61.5% filled-cell accuracy on the held-out evaluation set**.

You have NOT seen any held-out evaluation data. Final evaluation is **gated** — the user runs the benchmark themselves in a separate environment after you hand them a finalized model. You do not run it yourself, ever, and there is no script in this directory that would let you.

## Context

A previous OCR attempt was a 102K-parameter custom CNN trained on **synthetic data** (MNIST + ~4,500 printed digits from system fonts with augmentation + ~1,800 Chars74K held-out fonts + ~5,000 synthetic empty-cell variants). When evaluated zero-shot on the held-out evaluation set, that previous OCR scored:

- **61.5% filled-cell accuracy**
- **99.4% empty-cell accuracy** (very conservative — see failure mode)
- **5 perfect images** out of ~38 evaluable images
- **Failure mode**: 77% of errors are "missed" (CNN predicted empty when ground truth was filled). Only 6% are confidently wrong. The previous OCR is conservative on out-of-distribution inputs and abstains rather than commits.
- **Per-phone accuracy is bimodal**: very high on certain modern smartphones, near-zero on 2006-2008 phone cameras at high resolution. Failure is concentrated on specific phone/resolution combinations, not uniform across the held-out set.
- **Per-digit accuracy is non-uniform**: digits 8 and 9 are missed most often (~58-59% missed), digit 2 is best (~22% missed). Digits with denser ink suffer more.

The baseline you must beat: **61.5% filled-cell accuracy and 5 perfect images on the held-out evaluation set**.

## The hypothesis you're testing

1. **In-distribution training matters more than architecture choice.** The previous OCR's training data was synthetic and implicitly tuned to a different photo distribution. Your training data is real newspaper Sudoku cells from the **same camera/print distribution as the held-out set**. The hypothesis is that this matters more than any architectural improvement — that simply training on the right distribution will close most of the gap from 61.5% to a substantially higher number.
2. **Cell-extraction trimming is a free hyperparameter.** The previous OCR's preprocessing trimmed a 10% margin per side from each cell before resizing. This may not be enough — there's likely grid-line residue at cell borders that confuses the classifier. The cells in `data/train_cells/` are saved at the FULL 1/9 slice with no trim, so the trim percentage is your hyperparameter to tune. **This must be a required ablation in Phase 2 below.**

## Hard constraints

1. **Sandboxed working directory.** You may only read and write files inside the directory this notebook lives in (your cwd). Do not attempt to read any file via absolute paths outside it. There is no parent tree to peek at — the directory was copied here precisely so you cannot.
2. **No held-out set access.** A held-out evaluation set exists, but it is NOT in this directory and you must not seek it out. Do not try to obtain it via the network, via pip packages, via the upstream source dataset, or via any other channel. If you need to estimate generalization during development, use **k-fold cross-validation on the train cells you have** — that is your only allowed signal until the user runs the gated final evaluation themselves in a separate environment.
3. **No reuse of any prior model code.** You don't have access to it from this directory anyway. v2 is a clean reimplementation. You may use any third-party library (PyTorch, scikit-learn, ONNX, etc.) — anything `pip install`-able is fair game.
4. **Self-contained.** Every file your final model needs to load and run lives inside this directory. The user will be able to `cp -r` it back to the main repo and have everything work.
5. **Lab journal required.** Maintain `LAB_NOTES.md` with every experiment. Each entry needs a hypothesis, prediction, result, and one-sentence learning. This is auditable — the user reviews it after you finish.
6. **License.** The training images are CC BY 4.0. Attribute appropriately in any derivative; the literature-review PDFs in `research/` are the canonical citation source.

## Workflow — follow in order

### Phase 1 — Literature review (~1-2 hours)

Read every PDF in `research/`. For each, write a short summary in `research/summaries/<filename>.md` covering: **method, architecture, training data, reported results, what's reusable, what's obsolete**. The Wicht 2014 paper is the canonical baseline — his V1 dataset is closely related to your training set, and he reported 87.5% perfect-image rate on V1 test using a Deep Belief Network. His Ph.D. thesis reportedly hits 82.5% on V2 test with the same family of methods.

After the literature review, write **`GAME_PLAN.md`** listing 3-5 candidate approaches with tradeoffs. Examples to consider:

- **Modern small CNN** properly hyperparameter-tuned for this distribution (different from the previous OCR's architecture)
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
| 10% (matches the previous OCR) | 64% |
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

This is the same interface the user's existing pipeline uses, so v2 could be a drop-in replacement (the user does that integration after final eval, not you).

Save the trained weights to `checkpoints/v2.{pth,onnx}` (whichever your model uses). Write `RESULTS.md` summarizing:

1. Cross-validation accuracy on train set (your internal estimate)
2. The chosen architecture in 3-5 sentences (what / why / trained on / parameter count)
3. The trim ablation result from Phase 2
4. What you'd try next if you had another day

Then notify the user. **Do not run any held-out evaluation yourself** — that's the gate the user uses, in a separate environment outside this directory.

## Success criteria (in priority order)

1. **Cross-validation accuracy on train ≥ 75% filled-cell accuracy.** Below this, your model isn't learning anything in-distribution and there's no point running the test eval.
2. **Beats the previous OCR's 61.5% filled-cell accuracy** on the held-out evaluation set (the user verifies this themselves in a separate environment). This is the headline metric.
3. **Beats the previous OCR's 5 perfect images** on the held-out set (same gated user verification).
4. **Handles low-quality phone photos.** A meaningful share of held-out images come from older phone cameras at high resolution. The previous OCR essentially fails on these (under 40% filled accuracy on that subset). The training set you're given includes 23 images from one of those phone models, so a successful v2 should not be 0% on that distribution.
5. **Clean failure mode**: either conservative (low hallucinations) or roughly symmetric (wrong ≈ missed). Avoid models with many hallucinations — a single hallucinated digit kills Sudoku solvability.
6. **Explicable**: `LAB_NOTES.md`'s final entry should answer "why does this work and the previous OCR doesn't" in one paragraph.

## Anti-success — what NOT to do

- **Do not train on held-out data, even accidentally.** Cross-validate on train ONLY.
- **Do not import any prior model code.** Do not look for it. There's nothing in this directory to find.
- **Do not build an 800K-parameter monster.** The previous OCR is 102K params at ~40 ms per 81-cell batch. Stay in the same ballpark — under 1M params, under 200 ms per batch. This is a Sudoku OCR, not an OCR for the entire English language.
- **Do not skip the lab journal.** A great model with no journal is rejected. A mediocre model with a complete journal is accepted.
- **Do not skip Phase 2's trim ablation.** This is a required experiment. If it turns out to be a wash, that's a finding worth recording — but you must run it.
- **Do not build an ensemble that includes the user's previous OCR model.** You don't have access to it, and that would defeat the entire purpose of testing the in-distribution-training hypothesis.

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

Plus the user runs the gated held-out evaluation themselves in a separate environment to produce the official benchmark numbers.

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
