import subprocess

import numpy as np
import pandas as pd
import sentencepiece as spm

from pathlib import Path
from scipy.spatial import distance

def make_all_clusters_df(langs_dir):
    clustered_parquets = Path(f"/workspace/data/artefacts/ATDS/embeddings_xlsr-128_clustered/{langs_dir}").glob("*embeddings-clustered.parquet")

    all_clusters_df = pd.concat([
        pd.read_parquet(p).assign(lang=p.name.split("_")[0]) for p in
        clustered_parquets
    ])

    # Skip first 34 (reserved) chars of unicode table
    char_offset = 34
    all_clusters_df["cluster_char"] = [ chr(i + char_offset) for i in all_clusters_df.cluster_id ]
    
    return all_clusters_df

def make_all_utts_df(all_clusters_df):
    import re

    def merge_duplicates(line):
        return re.sub(r"(.)\1+", r"\1", line, 0, re.MULTILINE)

    all_utts_df = all_clusters_df.groupby(["lang", "wav_file"])["cluster_char"].apply(''.join).reset_index()
    all_utts_df.cluster_char = all_utts_df.cluster_char.apply(merge_duplicates)

    return all_utts_df

def train_and_encode_spm(all_utts_df, target_lang, ident_norm=False):
    with open('/workspace/tmp/tgt_utts.txt', 'w') as w1, open('/workspace/tmp/all_utts.txt', 'w') as w2:
        # For training sentencepiece model only on target language...
        w1.writelines("\n".join(all_utts_df[ all_utts_df.lang == target_lang ].cluster_char.to_list()) + "\n")

        # For inference to get sentencepiece IDs for all languages using trained model
        w2.writelines("\n".join(all_utts_df.cluster_char.to_list()) + "\n")
        
    spm_args = [
        "spm_train",
        "--input=/workspace/tmp/tgt_utts.txt",
        "--model_prefix=/workspace/tmp/10k_piece",
        "--vocab_size=10001",
        "--character_coverage=1",
        "--model_type=unigram",
        "--bos_id=-1",
        "--eos_id=-1"
    ]
    
    if ident_norm is True:
        # Prevent spm normalisation (on by default) if we need to recover mappings
        # to compute input-output alignments for TextGrids
        spm_args += ["--normalization_rule_name", "identity"]
    
    # Train sentencepiece model on target data
    subprocess.run(spm_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    # Load trained model
    s = spm.SentencePieceProcessor(model_file='/workspace/tmp/10k_piece.model')
    
    # Encode all utterances
    all_utts_df["utt_piece_ids"] = all_utts_df.cluster_char.apply(lambda x: s.encode(x, out_type=int))
    
    return all_utts_df

def make_piece_freqs_matrix(all_utts_df, target_lang):
    piece_counts_matrix = all_utts_df \
        .explode('utt_piece_ids')[['lang', 'utt_piece_ids']] \
        .groupby('lang')['utt_piece_ids'] \
        .value_counts() \
        .to_frame('count') \
        .reset_index() \
        .pivot(index='utt_piece_ids', columns='lang', values='count') \
        .fillna(0)
    
    # Normalize counts in each language column to most frequent piece
    for c in piece_counts_matrix.columns:
        piece_counts_matrix[c] /= piece_counts_matrix[c].max()

    return piece_counts_matrix[[target_lang] + [ c for c in piece_counts_matrix.columns if c != target_lang ]]

def make_ATDS_matrix(piece_freqs_matrix):
    langs = list(piece_freqs_matrix.columns)

    dists_df = pd.DataFrame([], columns=["ref_lang"] + langs)

    for r in langs:

        row_data = {"ref_lang":r}

        for c in langs:
            if r == c:
                row_data[c] = [None]
            else:
                row_data[c] = [ 1 - distance.cosine(piece_freqs_matrix[r].to_list(), piece_freqs_matrix[c].to_list()) ]

        dists_df = dists_df.append(pd.DataFrame(row_data))

    return dists_df

def get_best_donors_by_ATDS(ATDS_matrix, target_lang):
    atds_df = ATDS_matrix.rename(columns={'ref_lang':'target_lang'}) \
        .melt(id_vars=['target_lang'], var_name="donor_lang", value_name="atds") \
        .sort_values('target_lang') \
        .query(f"target_lang=='{target_lang}' and target_lang!=donor_lang") \
        .sort_values('atds', ascending=False) \

    atds_df.atds = atds_df.atds.apply(lambda x: round(x, 2))

    return atds_df

def run_all(langs_dir, target_lang, ident_norm=False):
    all_clusters_df = make_all_clusters_df(langs_dir)
    
    all_utts_df = make_all_utts_df(all_clusters_df)
    all_utts_df = train_and_encode_spm(all_utts_df, target_lang, ident_norm=ident_norm)
    
    piece_freqs_matrix = make_piece_freqs_matrix(all_utts_df, target_lang)
    
    atds_matrix = make_ATDS_matrix(piece_freqs_matrix)  
    best_donors = get_best_donors_by_ATDS(atds_matrix, target_lang)

    return atds_matrix, best_donors
