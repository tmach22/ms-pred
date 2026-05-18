#!/bin/bash
# Pull the latest Re-Ranker checkpoints from PVC to local.
# Usage: bash nrp/fetch-checkpoints.sh [local_output_dir]
# Default output: ./weights/ms3_reranker/
set -e

NS="wang-bioinf"
PVC="ms-pred-softlabel-pvc"
LOCAL_OUT="${1:-./weights/ms3_reranker}"
SHUTTLE="ckpt-shuttle-$$"

mkdir -p "$LOCAL_OUT"

cleanup() { kubectl delete pod "$SHUTTLE" -n "$NS" --ignore-not-found --wait=false 2>/dev/null || true; }
trap cleanup EXIT

echo "Spinning up shuttle pod..."
kubectl run "$SHUTTLE" -n "$NS" \
  --image=busybox --restart=Never \
  --overrides="{
    \"spec\": {
      \"containers\": [{
        \"name\": \"$SHUTTLE\",
        \"image\": \"busybox\",
        \"command\": [\"sleep\", \"300\"],
        \"volumeMounts\": [{\"name\":\"pvc\",\"mountPath\":\"/workspace\"}]
      }],
      \"volumes\": [{\"name\":\"pvc\",\"persistentVolumeClaim\":{\"claimName\":\"$PVC\"}}]
    }
  }"

kubectl wait pod/"$SHUTTLE" -n "$NS" --for=condition=Ready --timeout=90s

echo "Copying checkpoints to $LOCAL_OUT ..."
# PVC is mounted at /workspace in the shuttle; .pt files are written to the PVC root
# (the training job mounts PVC at /workspace/results, so CHECKPOINT_DIR maps to PVC root)
kubectl exec -n "$NS" "$SHUTTLE" -- sh -c 'ls /workspace/*.pt 2>/dev/null || true' | \
  while read -r f; do
    fname=$(basename "$f")
    kubectl cp "$NS/$SHUTTLE:$f" "$LOCAL_OUT/$fname"
  done
# Also copy latest/best if present
for sentinel in reranker_latest.pt reranker_best.pt; do
  kubectl exec -n "$NS" "$SHUTTLE" -- sh -c "[ -f /workspace/$sentinel ] && echo yes || echo no" 2>/dev/null | \
    grep -q yes && kubectl cp "$NS/$SHUTTLE:/workspace/$sentinel" "$LOCAL_OUT/$sentinel" || true
done

echo ""
echo "Done. Checkpoints in: $LOCAL_OUT"
ls -lh "$LOCAL_OUT"/*.pt 2>/dev/null || echo "(no .pt files yet)"
