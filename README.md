Note: if you are viewing this repository on https://anonymous.4open.science/r/2FF2, you will see `XXXX-LINENO` in places where there is identifying information.

# Predicting positive transfer for improved low-resource speech recognition using acoustic pseudo-tokens

## About

In this project, we are interested in 1) whether mixing in data from a similar, higher-resource 'donor' language during continued pre-training of wav2vec 2.0 XLSR-128 helps improve downstream word error rate on a target language and 2) how we can predict how (un)helpful various donor languages are for a given target language.

For doing such predictions, we propose the Acoutic Token Distribution Similarity (ATDS) and show that the ATDS between the target language and its candidate donors precisely predicts target language ASR performance across a set of typologically different target languages (Punjabi, Galician, Iban, Setswana).

This repository contains all code and configuration files for reproducing these results.

## Data

### Speech data

Training data manifests for both continued pre-training and fine-tuning are contained in the `data/manifests` directory.
The corresponding audio (several hundred gigabytes) should be sourced from the original corpora: 
- IndicSUPERB: https://ai4bharat.iitm.ac.in/indicsuperb/
- CommonVoice v15.0 (Galician, Spanish, Portuguese, Indonesian): https://commonvoice.mozilla.org/en/datasets
- NCHLT (Setswana, Sesotho, Sepedi): https://repo.sadilar.org/
- Iban: https://openslr.magicdatatech.com/24/
- MASS (Malay): requires request to MASS corpus compiler ([Tan Tien Ping](https://cs.usm.my/index.php/faculty-member/203-tan-tien-ping-dr))

### Experiment artefacts

All experiment artefacts are available on Zenodo: https://zenodo.org/communities/w2v2-cpt-transfer.
Since each wav2vec 2.0 checkpoint is about 3.8 GB and there were 53 checkpoints for the Punjabi experiments alone, they are split across multiple records (which are limited to 50 GB each) and all records are listed as part of a Zenodo 'community'.

## Reproducibility

All analyses are reproducible from the experiment artefacts. Each analysis is a self contained notebook located in the analyses folder.

## Training your own models

### Data preparation

Since we're using the [fairseq](https://github.com/facebookresearch/fairseq) library for pre-training and fine-tuning, the data will need to be in the format expected by fairseq. The actual manifests used in our experiments are available as examples in `data/manifests/pretrain` and `data/manifests/finetune`.

### Environment

We provide a Docker image with all the necessary dependencies installed. To start the container, run:

```
docker-compose run w2v2-cpt-transfer
```

### Model training

To continue pre-training the XLSR-128 model, download the official checkpoint into the `checkpoints` folder:

```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -P checkpoints/
```

#### Pre-training

The two commands below show examples of how to start a pre-training run using our configuration file (located in `configs/w2v2-large-cpt_indic-70h.yaml`). As specified in the configuration file by the `task.data` parameter, it will assume the data are located in `/workspace/data/manifests/pretrain/`.

```bash
# Use a single dataset
fairseq-hydra-train \
    --config-dir /workspace/configs \
    --config-name w2v2-large-cpt_indic-70h
    dataset.train_subset='punjabi_train-10h'

# Use a mix of two datasets, evenly sampling from each
fairseq-hydra-train \
    --config-dir /workspace/configs \
    --config-name w2v2-large-cpt_indic-70h \
    dataset.train_subset='punjabi_train-10h,malayalam_train-60h-seed-1'
```

#### Fine-tuning

To fine-tune a wav2vec 2.0 model on Punjabi (which is the default fine-tuning dataset for `configs/w2v2-large-finetune_punjabi-1h.yaml`), run:

```bash
fairseq-hydra-train \
    --config-dir /workspace/configs \
    --config-name w2v2-large-finetune_punjabi-1h \
    model.w2v_path=/workspace/checkpoints/xlsr2_300m.pt

# To use a different dataset, over-ride the task.data location and train/valid subset names
fairseq-hydra-train \
    --config-dir /workspace/configs \
    --config-name w2v2-large-finetune_punjabi-1h \
    model.w2v_path=/workspace/checkpoints/xlsr2_300m.pt \
    task.data=/workspace/data/manifests/finetune/galician_1h \
    dataset.train_subset=train \
    dataset.valid_subset=valid
```

### ASR evaluation

We use the inference script within the `fairseq` repo to derive transcriptions for the test set from a fine-tuned model:

```bash
python /fairseq/examples/speech_recognition/infer.py \
    /workspace/data/manifests/finetune/punjabi \
    --gen-subset test-2h \
    --path /workspace/checkpoints/finetuned/punjabi/xlsr2_300m_ft-pa-1h.pt
    --results-path /workspace/data/artefacts/asr-results/punjabi \
    --task audio_finetuning \
    --nbest 1 \
    --w2l-decoder viterbi \
    --criterion ctc \
    --labels ltr \
    --max-tokens 5000000 \
    --post-process letter
```