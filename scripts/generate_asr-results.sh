for TARGET_LANG in "punjabi" "galician" "iban" "setswana"; do

    MANIFEST_DIR=/workspace/data/manifests/finetune/${TARGET_LANG}
    
    TEST_MANIFEST_TSV=$(set -- "$MANIFEST_DIR/test*.tsv"; echo $1)
    TEST_FILENAME=$(basename "$TEST_MANIFEST_TSV" .tsv)

    RESULTS_PATH=/workspace/data/artefacts/asr-results/${TARGET_LANG}

    mkdir -p $RESULTS_PATH

    for CP_FILE in checkpoints/finetuned/${TARGET_LANG}/*.pt; do

        python /fairseq/examples/speech_recognition/infer.py \
            /workspace/data/manifests/finetune/${TARGET_LANG} \
            --gen-subset ${TEST_FILENAME} \
            --path ${CP_FILE} \
            --results-path ${RESULTS_PATH} \
            --task audio_finetuning \
            --nbest 1 \
            --w2l-decoder viterbi \
            --criterion ctc \
            --labels ltr \
            --max-tokens 5000000 \
            --post-process letter

    done

done
