nextflow.enable.dsl=2

params.stage1_dir   = "/workspace/data/MSnLib/processed_stage1"
params.master_index = "/workspace/data/MSnLib/master_metadata_index.parquet"
params.outdir       = "/workspace/data/MSnLib/distillation_tensors"
params.script       = "/workspace/code/fiar_pipeline/etl/prepare_distillation_tensors.py"

process PrepareTensors {
    publishDir "${params.outdir}", mode: 'copy'
    cpus 1
    memory '4 GB'

    input:
    path spectra_parquet
    path master_index

    output:
    path "distillation_tensors_*.parquet", emit: tensors optional true

    script:
    """
    python3 ${params.script} --spectra "${spectra_parquet}" --index "${master_index}" --outdir .
    """
}

workflow {
    spectra_ch = Channel.fromPath("${params.stage1_dir}/clean_spectra_*.parquet")
    index_ch   = file(params.master_index)
    PrepareTensors(spectra_ch, index_ch)
}
