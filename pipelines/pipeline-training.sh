#!/usr/bin/bash

if [ $# -lt 4 ]
then
    echo "$0 image_dir ocr_config db_file work_dir [stage]"
    exit 2
fi

img_dir=$1
config_ocr=$2
db_file=$3
work_dir=$4
mkdir -p "$work_dir"

stage=0
if [ $# -ge 5 ]
then
    stage=$5
fi

echo "$0 $img_dir $stage"

ocr_xml_dir="$work_dir/xml"
logits_dir="$work_dir/logits"
transcriptions_file="$work_dir/trancription"

if [ "$stage" -le 2 ]
then
    echo "== Stage 2 =="
    mkdir -p $ocr_xml_dir
    mkdir -p $logits_dir

    parse-folder \
        -c $config_ocr \
        -i $img_dir \
        --output-xml-path $ocr_xml_dir \
        --output-logit-path $logits_dir \
        --output-transcriptions-file-path $transcriptions_file \
        --device gpu \
        2>&1 | tee $ocr_xml_dir/parse_folder.log || exit 1
fi

db_pickle=$work_dir/db.pickle

if [ "$stage" -le 3 ]
then
    echo "== Stage 3 =="
    preprocess-library-db $db_pickle < "$db_file" || exit 1
fi

db_index_dir=$work_dir/index

if [ "$stage" -le 4 ]
then
    echo "== Stage 4 =="
    build-db-index \
        --bib-pickle $db_pickle \
        --index-dir $db_index_dir || exit 1

    cd $ocr_xml_dir
    for f in *.xml
    do
        cat $f | grep 'Unicode' | sed 's/^\s*<Unicode>//;s/<\/Unicode>$//' > $f.txt
    done
    cd ~-
fi

match_file=$work_dir/mapping.txt

if [ "$stage" -le 5 ]
then
    echo "== Stage 5 =="
    match-cards-to-db \
        --logging-level INFO \
        --index-dir $db_index_dir \
        --card-dir $ocr_xml_dir \
        --bib-pickle $db_pickle \
        --min-matched-lines 1 \
        > $match_file || exit 1
fi

cleaned_db_dir=$work_dir/cleaned-db
mkdir $cleaned_db_dir

if [ "$stage" -le 6 ]
then
    echo "== Stage 6 =="
    preprocess-alignment \
        --db-file $db_file \
        --mapping $match_file \
        --output $cleaned_db_dir || exit 1
fi

alignment_dir=$work_dir/alignments
alignment_all=$work_dir/alignment_all.txt

if [ "$stage" -le 7 ]
then
    echo "== Stage 7 =="
    align-records \
        --db-record $cleaned_db_dir \
        --transcription $ocr_xml_dir \
        --mapping $match_file \
        --output $alignment_dir \
        --threshold 0.8 \
        --max-candidates 5 || exit 1

    cat $alignment_dir/* > $alignment_all
fi

train_test=$work_dir/alignment_full.txt
if [ "$stage" -le 8 ]
then
    echo "== Stage 8 =="
    postprocess-alignment \
        --alignments-dir $alignment_dir \
        --output $train_test \
        || exit 1

fi


bert_path=/mnt/matylda5/ibenes/projects/pero/MZK-sw/work/bert-base-multilingual-uncased-sentiment

checkpoint_dir=$work_dir/checkpoints

mkdir -p $checkpoint_dir

pretrained_url=https://huggingface.co/bert-base-cased
pretrained_model_dir=$work_dir/pretrained_model

if [ "$stage" -le 9 ]
then
    echo "== Stage 9 == [This stage requires internet access]"
    git lfs clone $pretrained_url $pretrained_model_dir
fi

ner_model=$work_dir/ner_model
if [ "$stage" -le 10 ]
then
    echo "== Stage 10 == [This stage requires GPU access]"
    export TRANSFORMER_OFFLINE=1
    train-aligner \
        --epochs 5 \
        --batch-size 32 \
        --lr 3e-5\
        --sep \
        --bert-path $bert_path \
        --tokenizer-path $bert_path \
        --save-path $ner_model \
        --save-tokenizer \
        --ocr-path $ocr_xml_dir \
        --train-path $train_test \
        --val-path $train_test \
        --test-path $train_test \
        || exit 1

    ln -s $ner_model/checkpoint_005.pth $ner_model/checkpoint_final.pth
fi
