nextflow.enable.dsl=2

// ── Parameters ────────────────────────────────────────────────────────────────
// Override on the command line with --raw_dir, --outdir, or --script
params.raw_dir = "${projectDir}/../../data/MSnLib/raw_scans/positive"
params.outdir  = "${projectDir}/../../data/MSnLib/processed_stage1"
params.script  = "${projectDir}/../etl/tensorizer.py"

// ── Process ───────────────────────────────────────────────────────────────────
process Tensorize {
    publishDir "${params.outdir}", mode: 'copy'

    cpus   2
    memory '8 GB'

    input:
    path mzml_file

    output:
    path "clean_spectra_*.parquet", emit: parquets, optional: true
    path "*.log",                   emit: logs

    script:
    """
    python3 ${params.script} \
        --mzml   "${mzml_file}" \
        --outdir . \
        > "${mzml_file.baseName}_tensorizer.log" 2>&1
    """
}

// ── Workflow ──────────────────────────────────────────────────────────────────
workflow {
    mzml_ch = Channel.fromPath("${params.raw_dir}/*.mzML")
    Tensorize(mzml_ch)
}
