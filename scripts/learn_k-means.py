import joblib
import sys

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class EmbedsForKMeans(Dataset):
 
  def __init__(self, embeddings_parquet):

    embedding_cols = [ f"e{i:03}" for i in range(1024) ]
    self.embeddings_df = pd.read_parquet(embeddings_parquet, columns=embedding_cols)
 
  def __len__(self):
    return len(self.embeddings_df)
   
  def __getitem__(self,idx):
    return np.array(self.embeddings_df.iloc[idx])

input_file = sys.argv[1]
output_file = sys.argv[2]

ds = EmbedsForKMeans(input_file)
dl = DataLoader(ds, batch_size=10_000, shuffle=True)
di = iter(dl)

MAX_STEPS=500

km_model = MiniBatchKMeans(
    n_clusters=500,
    init="k-means++",
    max_iter=MAX_STEPS,
    batch_size=10_000,
    verbose=0,
    compute_labels=False,
    tol=0.0,
    max_no_improvement=100,
    init_size=None,
    n_init=20,
    reassignment_ratio=0.0,
)

for i in tqdm(range(MAX_STEPS)):

    if i >= MAX_STEPS:
        break

    try:
        batch = next(di)
    except StopIteration:
        # StopIteration is thrown if dataset ends, reset for new epoch
        di = iter(dl)
        batch = next(di)

    km_model.partial_fit(batch)
    
joblib.dump(km_model, output_file)
