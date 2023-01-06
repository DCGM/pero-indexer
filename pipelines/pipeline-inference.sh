#!/usr/bin/bash

if [ $# -lt 4 ]
then
    echo "$0 image_dir bert_dir ocr_config work_dir [stage]"
    exit 2
fi

img_dir=$1
bert_dir=$2
config_ocr=$3
work_dir=$4
mkdir -p "$work_dir"

stage=0
if [ $# -ge 5 ]
then
    stage=$5
fi

echo "$0 $img_dir $bert_dir $work_dir $stage"

ocr_xml_dir="$work_dir/xml"
logits_dir="$work_dir/logits"
transcriptions_file="$work_dir/trancription"

local_txt_dir="texts"
ocr_txt_dir="$work_dir/$local_txt_dir"

if [ "$stage" -le 1 ]
then
    mkdir -p $ocr_xml_dir
    mkdir -p $logits_dir

    parse-folder \
        -c "$config_ocr" \
        -i "$img_dir" \
        --output-xml-path "$ocr_xml_dir" \
        --output-logit-path "$logits_dir" \
        --output-transcriptions-file-path "$transcriptions_file" \
        --device gpu \
        2>&1 | tee "$ocr_xml_dir/parse_folder.log" || exit 1

    mkdir -p $ocr_txt_dir

    cd $ocr_xml_dir
    for f in *.xml
    do
        cat $f | grep 'Unicode' | sed 's/^\s*<Unicode>//;s/<\/Unicode>$//' > "../$local_txt_dir/$f.txt"
    done
    cd ~-
fi

final_output="$work_dir/output"

if [ "$stage" -le 2 ]
then
    export TRANSFORMER_OFFLINE=1
    run-aligner \
        --data-path "$ocr_txt_dir" \
        --config-path "$bert_dir" \
        --model-path "$bert_dir/checkpoint_final.pth" \
        --tokenizer-path "$bert_dir" \
        --save-path "$final_output" \
        || exit 1
fi

readable_output="$work_dir/readable_output"

if [ "$stage" -le 3 ]
then
    get-readable-output \
        --dataset "$final_output/dataset.all" \
        --ocr "$ocr_txt_dir" \
        --out "$readable_output" \
        || exit 1
fi
