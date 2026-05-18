#!/bin/bash
# Designed for crontab — no interactive auth, uses saved kubeconfig.
# Add to crontab: */30 * * * * bash /home/user/ms-pred/nrp/cron-fetch-checkpoints.sh >> /tmp/ckpt-fetch.log 2>&1

export PATH="/home/user/miniforge3/envs/kube/bin:$PATH"
KUBECONFIG="$HOME/.kube/config"
NS="wang-bioinf"
PVC="ms-pred-softlabel-pvc"
LOCAL_OUT="/home/user/ms-pred/weights/ms3_reranker"
SHUTTLE="ckpt-cron-$$"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

mkdir -p "$LOCAL_OUT"

cleanup() {
    kubectl --kubeconfig="$KUBECONFIG" delete pod "$SHUTTLE" -n "$NS" \
        --ignore-not-found --wait=false 2>/dev/null || true
}
trap cleanup EXIT

# Spin up shuttle
kubectl --kubeconfig="$KUBECONFIG" run "$SHUTTLE" -n "$NS" \
  --image=busybox --restart=Never \
  --overrides="{
    \"spec\":{
      \"containers\":[{
        \"name\":\"$SHUTTLE\",
        \"image\":\"busybox\",
        \"command\":[\"sleep\",\"120\"],
        \"volumeMounts\":[{\"name\":\"pvc\",\"mountPath\":\"/workspace\"}]
      }],
      \"volumes\":[{\"name\":\"pvc\",\"persistentVolumeClaim\":{\"claimName\":\"$PVC\"}}]
    }
  }" 2>&1 || { echo "$LOG_PREFIX ERROR: failed to create shuttle pod"; exit 1; }

kubectl --kubeconfig="$KUBECONFIG" wait pod/"$SHUTTLE" -n "$NS" \
    --for=condition=Ready --timeout=90s 2>&1 || { echo "$LOG_PREFIX ERROR: shuttle pod never Ready"; exit 1; }

# Copy all .pt files from PVC root
FOUND=0
for remote_pt in $(kubectl --kubeconfig="$KUBECONFIG" exec -n "$NS" "$SHUTTLE" -- \
        sh -c 'ls /workspace/*.pt 2>/dev/null' 2>/dev/null); do
    fname=$(basename "$remote_pt")
    kubectl --kubeconfig="$KUBECONFIG" cp "$NS/$SHUTTLE:$remote_pt" "$LOCAL_OUT/$fname" 2>/dev/null
    FOUND=$((FOUND + 1))
done

# Also grab latest/best sentinels
for sentinel in reranker_latest.pt reranker_best.pt; do
    exists=$(kubectl --kubeconfig="$KUBECONFIG" exec -n "$NS" "$SHUTTLE" -- \
        sh -c "[ -f /workspace/$sentinel ] && echo yes || echo no" 2>/dev/null)
    if [ "$exists" = "yes" ]; then
        kubectl --kubeconfig="$KUBECONFIG" cp "$NS/$SHUTTLE:/workspace/$sentinel" \
            "$LOCAL_OUT/$sentinel" 2>/dev/null
        FOUND=$((FOUND + 1))
    fi
done

echo "$LOG_PREFIX Fetched $FOUND .pt file(s)."
ls -lh "$LOCAL_OUT"/*.pt 2>/dev/null || echo "$LOG_PREFIX (no .pt files in $LOCAL_OUT)"
