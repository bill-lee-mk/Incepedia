# Detachment & Watchdog Infrastructure

Survives cursor session restart / SSH drop / agent timeout.

## 4 components

| File | Purpose |
|------|---------|
| `scripts/run_detached.sh` | TRUE daemonization (verifies PPID==1 before returning) |
| `scripts/train_watchdog.py` | health checks + auto-resume; cron-friendly |
| `scripts/test_detach.sh` | validates run_detached.sh works (run once before any new training) |
| `configs/watchdog/jobs.json` | list of jobs the watchdog polls (tracked in git) |

## Failure modes the watchdog detects

1. **Process dead, training not finished** → relaunch
2. **Process alive but log stale (>30 min)** → kill + relaunch
3. **Iteration counter not advancing 3+ polls (=15 min stuck)** → kill + relaunch
4. **Process clean-exited with metrics.json present** → no action (training finished)

## Launching a new training (the new way)

```bash
# Step 1: smoke test detachment infrastructure
bash scripts/test_detach.sh

# Step 2: launch training detached
bash scripts/run_detached.sh protC_train -- \
    /home/ubuntu/miniconda3/envs/incepedia/bin/python \
    scripts/run_experiment.py \
    --config experiments/exp_finephrase_repro_protC_seed42/config.yaml

# Step 3: register with watchdog (already done in logs/detached/watchdog.json)

# Step 4: verify cron polls it
crontab -l | grep train_watchdog
tail -F logs/detached/watchdog.log  # 5-min poll log
tail -F logs/detached/protC_train.log  # actual training log
```

## Inspecting / killing

```bash
# Status
pid=$(cat logs/detached/protC_train.pid)
ps -p $pid -o pid,ppid,sid,etime,pcpu,pmem,cmd

# Tail training
tail -F logs/detached/protC_train.log | grep iteration:

# Watchdog state
cat logs/detached/watchdog_protC_train.state.json | python3 -m json.tool

# Kill (DO NOT do this if you want auto-resume; instead delete pid file
# then kill, otherwise watchdog will relaunch in <5min)
rm logs/detached/protC_train.pid
kill $pid
```

## Lessons learned (why this exists)

The 2026-04-24 training run died at iter 1411/10013 because:
1. We used cursor's `block_until_ms: 0` thinking it was true detachment —
   it wasn't (the process was still a child of cursor's bash wrapper).
2. When cursor session ended (~3.3h after launch), SIGHUP propagated,
   killing nanotron with no traceback.
3. We had no watchdog, so 37 hours of GPU were wasted before anyone
   noticed the training had died.
4. By the time we discovered it, another user had reasonably claimed
   the GPU, and we couldn't kill their job (AGENTS.md §3).

This infrastructure prevents (1)+(2) by enforcing PPID==1 verification
before declaring "detached", and prevents (3) via cron polling.
