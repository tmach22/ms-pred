#!/usr/bin/env bash
# =============================================================================
# MSnLib Dataset Downloader (Python/No-JQ Edition)
# =============================================================================
# Downloads:
#   Target 1 — Processed JSON libraries   (Zenodo 11163380)
#   Target 2 — Raw mzML scans pos + neg   (Zenodo 10966280)
#   Target 3 — MassIVE mirror (optional)  (MSV000094528)
#
# Requires: wget, curl, python3
# Resume:   All wget calls use -c (continue interrupted downloads).
# Usage:    bash download_msnlib.sh [--skip-massive]
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_ROOT="/home/user/ms-pred/data/MSnLib"
ZENODO_LIBS_ID="11163380"   # processed .mgf + .json libraries
ZENODO_MZML_ID="10966280"   # raw .mzML pos + neg
MASSIVE_ACC="MSV000094528"

SKIP_MASSIVE=false
if [[ "${1:-}" == "--skip-massive" ]]; then SKIP_MASSIVE=true; fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p \
    "${TARGET_ROOT}/libraries/json" \
    "${TARGET_ROOT}/libraries/mgf" \
    "${TARGET_ROOT}/raw_scans/positive" \
    "${TARGET_ROOT}/raw_scans/negative" \
    "${TARGET_ROOT}/massive"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Dependency check ──────────────────────────────────────────────────────────
for cmd in wget curl python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[!] Required tool not found: $cmd"
        exit 1
    fi
done

# ── Helper: Extract files from Zenodo API using pure Python ───────────────────
get_zenodo_files() {
    local record_id="$1"
    # Note: Zenodo InvenioRDM embeds files in the main record metadata
    curl -fsSL "https://zenodo.org/api/records/${record_id}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    files = data.get('files', [])
    if isinstance(files, dict):
        files = files.get('entries', files.values())
    for f in files:
        key = f.get('key') or f.get('filename') or ''
        size = f.get('size', 0)
        links = f.get('links', {})
        url = links.get('self') or links.get('content') or links.get('download') or ''
        print(f'{key}\t{url}\t{size}')
except Exception as e:
    pass
"
}

# ── Target 1: Processed JSON + MGF libraries ─────────────────────────────────
log "=== Target 1: Processed Libraries (Zenodo ${ZENODO_LIBS_ID}) ==="
get_zenodo_files "${ZENODO_LIBS_ID}" | while IFS=$'\t' read -r filename url remote_size; do
    [[ -z "${filename}" || -z "${url}" ]] && continue

    case "${filename}" in
        *.json) dest_dir="${TARGET_ROOT}/libraries/json" ;;
        *.mgf)  dest_dir="${TARGET_ROOT}/libraries/mgf"  ;;
        *)      dest_dir="${TARGET_ROOT}/libraries"      ;;
    esac

    dest_file="${dest_dir}/${filename}"
    local_size=$(stat -c%s "${dest_file}" 2>/dev/null || echo 0)

    if [[ "${local_size}" == "${remote_size}" ]]; then
        log "  SKIP (complete): ${filename}"
        continue
    fi

    log "  GET [${dest_dir##*/}]: ${filename}"
    wget -c -q --show-progress -O "${dest_file}" "${url}" \
    && log "  OK: ${filename}" \
    || log "  WARN: wget exited non-zero for ${filename}"
done

# ── Target 2: Raw mzML scans (pos + neg) ─────────────────────────────────────
log "=== Target 2: Raw mzML Scans (Zenodo ${ZENODO_MZML_ID}) ==="
get_zenodo_files "${ZENODO_MZML_ID}" | while IFS=$'\t' read -r filename url remote_size; do
    [[ -z "${filename}" || -z "${url}" ]] && continue

    # Only process .mzML files
    [[ "${filename}" != *.mzML ]] && [[ "${filename}" != *.mzml ]] && continue

    if echo "${filename}" | grep -qiE '(pos|positive)'; then
        dest_dir="${TARGET_ROOT}/raw_scans/positive"
    elif echo "${filename}" | grep -qiE '(neg|negative)'; then
        dest_dir="${TARGET_ROOT}/raw_scans/negative"
    else
        dest_dir="${TARGET_ROOT}/raw_scans"
    fi

    dest_file="${dest_dir}/${filename}"
    local_size=$(stat -c%s "${dest_file}" 2>/dev/null || echo 0)

    if [[ "${local_size}" == "${remote_size}" ]]; then
        log "  SKIP (complete): ${filename}"
        continue
    fi

    log "  GET [$(basename ${dest_dir})]: ${filename}"
    wget -c -q --show-progress -O "${dest_file}" "${url}" \
    && log "  OK: ${filename}" \
    || log "  WARN: wget exited non-zero for ${filename}"
done

# ── Target 3: MassIVE mirror (opt-in) ─────────────────────────────────────────
if [[ "${SKIP_MASSIVE}" == false ]]; then
    log "=== Target 3: MassIVE mirror (${MASSIVE_ACC}) ==="
    log "  Connecting to MassIVE FTP — this may take a long time."
    wget -r -c -q --show-progress --no-parent --no-host-directories --cut-dirs=1 \
         -A "*.mzML,*.mzml" -P "${TARGET_ROOT}/massive" \
         "ftp://massive.ucsd.edu/${MASSIVE_ACC}/" \
    && log "MassIVE download complete." \
    || log "WARN: MassIVE wget exited non-zero."
fi

log "Done. Data root: ${TARGET_ROOT}"