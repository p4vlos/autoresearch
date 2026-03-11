# autoresearch

This is an experiment to have the LLM do its own research.

## Domain: GLOBEM Depression Prediction

**Task:** Binary depression prediction from 28-day mobile sensing windows.
- **Input:** `(28, N_features)` — 28 days of daily sensor-derived features (location, screen, calls, bluetooth, steps, sleep)
- **Output:** Binary label — depressed (PHQ-4 > 2) or not
- **Metric:** Balanced accuracy (higher is better) — handles class imbalance (~40-46% positive)
- **Secondary metric:** ROC-AUC

The agent can experiment with: architectures (MLP, ConvNet, Transformer, LSTM), feature selection, data augmentation, class weighting strategies, regularization (dropout, weight decay), learning rate schedules, and model size.

## Infrastructure: SLURM + V100

Experiments run on an HPC cluster via SLURM. Each experiment is a single-GPU (V100) SBATCH job.

- **GPU:** V100 32GB (SM70, float16 only — no bfloat16)
- **Partition:** `gpu`, 1 GPU per job
- **Time limit:** 15 minutes per SBATCH job (5-min training + startup/eval overhead)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains cached `.pt` files. If not, tell the human to run `uv run prepare.py` (or `bash slurm/setup.sh` on the cluster).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single V100 GPU via SLURM. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**Launching an experiment:**
```bash
# Submit a SLURM job
sbatch slurm/job.sbatch
```

**Waiting for results:**
```bash
# Check if job is still running
squeue -u $USER

# Once finished, read the output
cat results/slurm-<jobid>.out
```

Note: SLURM queue wait time does NOT count toward the 5-minute training budget.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, window days, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest balanced_acc.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. The V100 has 32GB. Some increase is acceptable for meaningful balanced_acc gains, but it should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
balanced_acc:     0.5820
roc_auc:          0.6340
training_seconds: 300.1
total_seconds:    315.2
peak_vram_mb:     1024.5
num_steps:        4500
num_params_K:     125.3
```

You can extract the key metric from the SLURM output file:

```
grep "^balanced_acc:" results/slurm-<jobid>.out
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	balanced_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. balanced_acc achieved (e.g. 0.5820) — use 0.0000 for crashes
3. peak memory in GB, round to .1f (e.g. 1.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	balanced_acc	memory_gb	status	description
a1b2c3d	0.5820	1.0	keep	baseline MLP
b2c3d4e	0.6010	1.2	keep	increase hidden dim to 512
c3d4e5f	0.5750	1.0	discard	switch to GeLU activation
d4e5f6g	0.0000	0.0	crash	double model layers (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `sbatch slurm/job.sbatch` and wait for it to finish
5. Read out the results: `grep "^balanced_acc:\|^peak_vram_mb:" results/slurm-<jobid>.out`
6. If the grep output is empty, the run crashed. Run `tail -n 50 results/slurm-<jobid>.out` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If balanced_acc improved (higher), you "advance" the branch, keeping the git commit
9. If balanced_acc is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**SLURM debugging tips:**
- `squeue -u $USER` — check job status
- `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` — detailed job info
- `scancel <jobid>` — cancel a stuck job
- If a job fails with `NODE_FAIL` or `TIMEOUT`, it's a cluster issue — just resubmit
- SLURM output goes to `results/slurm-<jobid>.out` (both stdout and stderr)

**Timeout**: Each experiment should take ~5 minutes of training time + a few minutes for startup, compilation, and evaluation. The SBATCH time limit is 15 minutes. If a job exceeds that, SLURM kills it — treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try different architectures (ConvNet, LSTM, Transformer), feature engineering, augmentation strategies, ensemble methods. The loop runs until the human interrupts you, period.
