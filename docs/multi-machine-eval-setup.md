# Multi-machine eval setup — 164 (A100 box) reading from 165 (H100 train box)

Status: 2026-04-22 — written when bringing the dedicated A100 evaluation node online.

## TL;DR

| Step | Where | What |
|------|-------|------|
| 1 | Lambda Labs console | Attach the same shared volume (UUID `aa4b366d-e191-4f78-a42a-f817ce6b86ee`) to the A100 instance OR keep them separate |
| 2 | A100 box (164) | `bash scripts/mount_lambda_nas.sh attach` (if step 1 done) **or** `bash scripts/mount_lambda_nas.sh sshfs ubuntu@165` (fallback) **or** rely on rsync only (no live mount) |
| 3 | A100 box (164) | Clone the repo, run `bash scripts/bootstrap_env.sh` (env + nanotron + datatrove + patches) |
| 4 | H100 box (165) | After every training: `bash scripts/sync_to_nas.sh hf_ckpt <exp_id>` |
| 5 | A100 box (164) | `python scripts/run_experiment.py --config <...>/config.yaml --eval-only` reads `experiments/<exp_id>/hf_ckpt/` either from the live NAS mount or after a one-shot rsync from 165 |
| 6 | A100 box (164) | After eval: `bash scripts/sync_to_nas.sh eval <exp_id>` writes scores back |

## Why this is more involved than "just NFS-mount it"

Lambda Labs exposes the shared `/lambda/nfs/us-south-2` filesystem via **virtiofs**, not NFS. virtiofs is a hypervisor-level shared-memory filesystem (think QEMU/KVM virtio-fs) — it cannot be mounted from a sibling guest VM over the network. The only way to give 164 access to the same volume is to ask Lambda to attach the volume at the VM level (their console / API) — then virtiofs surfaces it identically.

If that is not possible (different region, billing, or just speed), you have two fallbacks documented in `scripts/mount_lambda_nas.sh`:

* **sshfs** — mount 165's NAS via SSH on 164. Works but throughput is SSH-encrypted, ~50–200 MB/s. Adequate for occasional eval (3.5 GB HF ckpt = ~30–60 s).
* **rsync only** — no live mount, transfer per-need (today's `scripts/sync_to_nas.sh` model).

## Recommended split of duties

| Box | Hardware | Role | Reads from NAS | Writes to NAS |
|------|----------|------|----------------|----------------|
| 165  | 8 × H100 80G | Training, conversion, generation | `data/datasets/`, `data/reference/` | `experiments/<exp>/ckpt/`, `experiments/<exp>/hf_ckpt/` |
| 164  | 8 × A100 40G | Evaluation, light inference smoke tests | `experiments/<exp>/hf_ckpt/`, `data/hf_cache/datasets/` (eval datasets) | `experiments/<exp>/eval/`, INDEX.parquet rows |

INDEX.parquet is the single source of truth — both boxes read/write rows for the experiments they handle and rsync it back.

## Capacity check

The A100 box (164) has 5.7 TB local disk. We need:

* HF ckpt copy of any experiment we are evaluating: ~3.5 GB each (transient — can delete after eval)
* HF dataset cache for eval suite: ~16 GB (one-shot, kept)
* incepedia repo + env: ~10 GB
* total typical working set: **< 50 GB** — plenty of headroom on a 5.7 TB disk.

The 1.7B Qwen3 model in bf16 needs:

* model weights: 3.4 GB / GPU
* KV cache, activations, framework overhead: ~10 GB / GPU
* **peak ~15 GB / GPU** — fits comfortably in 40 GB A100s with > 60 % memory headroom.

## Concrete first-time setup on 164

```bash
# (1) On 164 — clone + bootstrap
git clone https://github.com/bill-lee-mk/Incepedia.git ~/Incepedia
cd ~/Incepedia
bash scripts/bootstrap_env.sh           # creates conda env + nanotron + patches

# (2) Choose ONE of:
#   (a) virtiofs (after Lambda attach)
bash scripts/mount_lambda_nas.sh attach
#   (b) sshfs from 165
bash scripts/mount_lambda_nas.sh sshfs ubuntu@192.222.52.165
#   (c) rsync only — skip mount; just keep ssh agent forwarding configured

# (3) Pull eval dataset cache once (16 GB)
rsync -a 'ubuntu@192.222.52.165:~/lilei/projects/Incepedia/data/hf_cache/' ~/Incepedia/data/hf_cache/

# (4) Per-experiment workflow:
#    on 165 (after train + convert):
#      bash scripts/sync_to_nas.sh hf_ckpt exp_xxx
#    on 164:
#      rsync -a /lambda/nfs/.../experiments/exp_xxx/hf_ckpt/ experiments/exp_xxx/hf_ckpt/  # if no virtiofs
#      python scripts/run_experiment.py --config experiments/exp_xxx/config.yaml --eval-only
#      bash scripts/sync_to_nas.sh eval exp_xxx
#      python scripts/index_experiment.py add exp_xxx
```

## Open issues

* **NAS access from the eval box is gated on Lambda's volume-attach** — until that is done, every eval requires a one-shot rsync of the HF ckpt (small, ~30s for 3.5 GB on a fast link).
* `lighteval`/`accelerate` eval path currently works for HF Qwen2 ckpts; the lighteval-nanotron path is broken in 0.13 (multiple cascading bugs, tracked in TODO T1 follow-up). The converter (`scripts/convert_nanotron_qwen2_to_hf.py`) restores the standard path.
