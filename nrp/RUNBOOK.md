# NRP Nautilus — Re-Ranker Training Runbook

Covers everything needed to go from zero to a running training job on NRP. Follow in order on a fresh attempt.

---

## Architecture

**Frozen ICEBERG Actor** (beam search, top-50) → **13-feature fingerprint** → **MAGMa Oracle** (CPU pool scoring) → **MLP Critic** (MSE loss, trained).

CPU is the bottleneck (MAGMa scoring), not GPU. Request as many CPUs as the node allows (32 confirmed on V100 DGX nodes).

---

## Permanent cluster state (already done — don't redo)

| Resource | Name / Value |
|---|---|
| Namespace | `wang-bioinf` |
| PVC | `ms-pred-softlabel-pvc` (rook-cephfs, RWX, 20Gi, mounted at `/workspace/results`) |
| Docker image | `ghcr.io/tmach22/ms-pred-env:v1` (public, 11.2 GB, bakes full `ms-gen` conda env) |
| kubectl binary | `/home/user/miniforge3/envs/kube/bin/kubectl` |

Always prefix kubectl commands with:
```bash
export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"
```

---

## One-time setup (already complete — reference only)

### 1. Export and bake the conda environment

```bash
# Export local ms-gen env (no build hashes, for cross-platform compatibility)
conda env export --no-builds -n ms-gen | grep -v "^prefix:" > nrp/environment.yml

# Build Docker image from repo root
docker build -f nrp/Dockerfile -t ghcr.io/tmach22/ms-pred-env:v1 .

# Push to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u tmach22 --password-stdin
docker push ghcr.io/tmach22/ms-pred-env:v1

# Make package public: GitHub → Packages → ms-pred-env → Package settings → Public
```

The Dockerfile (`nrp/Dockerfile`) installs `build-essential` (needed for gcc to compile `algos2.c`) before creating the conda env.

### 2. Create the PVC

```bash
export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"
kubectl apply -f nrp/softlabel-results-pvc.yaml
kubectl get pvc ms-pred-softlabel-pvc -n wang-bioinf  # verify Bound
```

---

## Per-run process

### Step 1 — Commit and push all code changes

The job clones the GitHub repo fresh on every run. **Local changes that are not committed and pushed will not be picked up.**

```bash
git add fiar_pipeline/scripts/train_distillation.py  # or whatever changed
git commit -m "your message"
git push
```

Critical env-var pattern in `train_distillation.py` — all config must come from `os.environ.get()`:
```python
CKPT_PATH      = os.environ.get("CKPT_PATH",      "<local_fallback>")
TRAIN_PATH     = os.environ.get("TRAIN_PATH",     "<local_fallback>")
VAL_PATH       = os.environ.get("VAL_PATH",       "<local_fallback>")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "<local_fallback>")
CPU_WORKERS    = int(os.environ.get("CPU_WORKERS", "20"))
```

### Step 2 — Bump the job name

Edit `nrp/reranker-job.yaml`, increment the version suffix:
```yaml
metadata:
  name: ms-pred-reranker-v15   # ← bump this
```

Never reuse a name — Kubernetes will reject it if the old job still exists (even completed).

### Step 3 — Submit the job

```bash
export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"
kubectl apply -f nrp/reranker-job.yaml
```

### Step 4 — Verify it's running (not Pending)

```bash
export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"
kubectl get pod -n wang-bioinf -l job-name=ms-pred-reranker-v<N> -o wide
# STATUS should reach Running within ~2 min if image is cached on node
```

If it stays `Pending` for more than 5 minutes, check events:
```bash
kubectl describe pod -n wang-bioinf <pod-name>
```

Common `Pending` causes and fixes:
- `Insufficient cpu` — node is full; wait or reduce CPU request
- `Unschedulable: 1 node had taint...nautilus.io/reservation` — A10 nodes are tainted; the affinity list in the YAML already excludes tainted types; if all listed types are full, wait
- Wrong GPU type list — verify against what's actually running: `kubectl get pods -n wang-bioinf -o wide | grep Running`

### Step 5 — Monitor logs

```bash
export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"
kubectl logs -n wang-bioinf job/ms-pred-reranker-v<N> --tail=50 -f
```

Expected output after startup (~2–3 min for clone + pip install + data download):
```
=== CUDA sanity check ===
CUDA OK: Tesla V100-SXM2-16GB
...
Dataset loaded with 210,481 rows.
Dataset loaded with 26,309 rows.
[ICEBERGScalpel] Ready on cuda:0.
Epoch 01 | Step 0000 | MSE Loss: 0.xxxx | Avg MAGMa: 0.xxxx | Pred mean: 0.xxxx
```

Training throughput: ~1 step every ~90 sec (CPU-bound MAGMa scoring with 28 workers). Each epoch ≈ 1,645 steps ≈ ~40 hours. Full 20-epoch run ≈ 5 days (within the 5-day `activeDeadlineSeconds: 432000` limit).

### Step 6 — Fetch checkpoints

The job saves to `/workspace/results` (the PVC) at two cadences:
- **Every 200 steps**: `reranker_epoch{EE}_step{SSSSS}.pt`
- **Every epoch**: `reranker_latest.pt` and `reranker_best.pt` (if val loss improved)

Pull checkpoints locally at any time (job does not need to be stopped):
```bash
bash nrp/fetch-checkpoints.sh                  # saves to ./weights/ms3_reranker/
bash nrp/fetch-checkpoints.sh /some/other/dir  # custom output dir
```

---

## GPU affinity reference

The YAML targets these types (confirmed to have 16–32 allocatable CPUs and no reservation taints in `wang-bioinf`):

| GPU | VRAM | Max CPUs seen |
|---|---|---|
| Tesla-V100-SXM2-32GB | 32 GB | 76 |
| Tesla-V100-SXM2-16GB | 16 GB | 76 |
| NVIDIA-A10 | 24 GB | 124 |
| NVIDIA-A40 | 48 GB | 60 |
| NVIDIA-L40S | 48 GB | 60 |
| NVIDIA-L4 | 24 GB | 60 |
| NVIDIA-GeForce-RTX-3090 | 24 GB | 68 |
| NVIDIA-GeForce-RTX-2080-Ti | 11 GB | 76 |

**Do not target A100 or H200** — NRP default quota is zero; pods pend forever.

Many A10 nodes carry `nautilus.io/reservation=usra:NoSchedule`. The affinity list intentionally keeps A10 to let unaffected A10 nodes match, while reservation-tainted ones self-exclude.

---

## Key pitfalls (learned the hard way)

| Symptom | Root cause | Fix |
|---|---|---|
| Wrong paths inside pod | `train_distillation.py` used hardcoded NAS paths, not committed | Always commit + push before submitting job |
| `ModuleNotFoundError: fiar_pipeline` | Missing `PYTHONPATH=/tmp/ms-pred` | Prepend `PYTHONPATH=/tmp/ms-pred` to python command in YAML |
| gcc compile error for `algos2.c` | conda env has no compiler | `build-essential` in Dockerfile `apt-get install` |
| Pod stays Pending (reservation taint) | A10 nodes in usra reservation | Use broad affinity list; don't pin to A10-only |
| `admission webhook denied: block-nodename` | Used `nodeName` field | Use `nodeSelector` or `nodeAffinity` instead |
| Job rejected on apply | Reused old job name that still exists | Always bump version suffix |
| Image pull slow (20+ min) | Large image (11.2 GB) pulled cold | Same-cluster nodes cache it; subsequent runs on same node are fast |

---

## Cleanup after run completes

```bash
export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"

# Pull final checkpoints before deleting anything
bash nrp/fetch-checkpoints.sh

# Delete the completed job (pod auto-deletes via ttlSecondsAfterFinished: 86400)
kubectl delete job ms-pred-reranker-v<N> -n wang-bioinf

# Keep the PVC — it holds results and can be reused for the next run
# Only delete PVC if you're done with results AND have pulled everything locally:
# kubectl delete pvc ms-pred-softlabel-pvc -n wang-bioinf  # IRREVERSIBLE
```
