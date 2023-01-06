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

if [ "$stage" -le 1 ]
then
    echo "== Stage 1 =="
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

if [ "$stage" -le 2 ]
then
    echo "== Stage 2 =="
    preprocess-library-db $db_pickle < "$db_file" || exit 1
fi

db_index_dir=$work_dir/index

if [ "$stage" -le 3 ]
then
    echo "== Stage 3 =="
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

if [ "$stage" -le 4 ]
then
    echo "== Stage 4 =="
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

if [ "$stage" -le 5 ]
then
    echo "== Stage 5 =="
    preprocess-alignment \
        --db-file $db_file \
        --mapping $match_file \
        --output $cleaned_db_dir || exit 1
fi

alignment_dir=$work_dir/alignments
alignment_all=$work_dir/alignment_all.txt

if [ "$stage" -le 6 ]
then
    echo "== Stage 6 =="
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
train_data=$work_dir/alignment.train
valid_data=$work_dir/alignment.valid
test_data=$work_dir/alignment.test
if [ "$stage" -le 7 ]
then
    echo "== Stage 7 =="
    postprocess-alignment \
        --alignments-dir $alignment_dir \
        --output $train_test \
        || exit 1

    nb_tot_lines=$(wc -l < $train_test)
    nb_train_lines=$(/usr/bin/printf %d $(bc -l <<< "$nb_tot_lines * 0.8") 2>/dev/null)
    nb_val_lines=$(/usr/bin/printf %d $(bc -l <<< "$nb_tot_lines * 0.1") 2>/dev/null)

    head -n "$nb_train_lines" < $train_test >$train_data
    tail -n "$(( nb_val_lines * 2 ))" < $train_test | head -n "$nb_val_lines"  >$valid_data
    tail -n "$nb_val_lines" < $train_test >$test_data

fi

pretrained_url=https://huggingface.co/bert-base-multilingual-uncased
pretrained_model_dir=$work_dir/pretrained_model

if [ "$stage" -le 8 ]
then
    echo "== Stage 8 == [This stage requires internet access]"
    git lfs clone $pretrained_url $pretrained_model_dir || exit 1
fi

bert_path=$work_dir/bert-base-multilingual-uncased-sentiment
ner_model=$work_dir/ner_model

if [ "$stage" -le 9 ]
then
    echo "== Stage 9 == [This stage requires GPU access]"

    nb_epochs=8

    export TRANSFORMER_OFFLINE=1
    train-aligner \
        --epochs "$nb_epochs" \
        --batch-size 32 \
        --sep \
        --bert-path $pretrained_model_dir \
        --tokenizer-path $pretrained_model_dir \
        --save-path $ner_model \
        --save-tokenizer \
        --ocr-path $ocr_xml_dir \
        --train-path $train_data \
        --val-path $valid_data \
        --test-path $test_data \
        || exit 1

    last_checkpoint_name=$(printf checkpoint_%03d.pth $nb_epochs)

    ln -f -s $last_checkpoint_name $ner_model/checkpoint_final.pth
fi
