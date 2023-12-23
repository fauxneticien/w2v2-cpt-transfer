import joblib
import sys

import numpy as np
import pandas as pd

km_model = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

km_model = joblib.load(km_model)

lang_embeds_file = input_file
print(f"Reading {lang_embeds_file}... ")

embedding_cols = [ f"e{i:03}" for i in range(1024) ]

lang_df = pd.read_parquet(
    lang_embeds_file,
    columns=["wav_file"] + embedding_cols
)

lang_df["cluster_id"] = km_model.predict(np.array(lang_df[embedding_cols]).astype(float))

lang_df[ ["wav_file", "cluster_id"] ].to_parquet(output_file)
