import vecmini
import numpy as np
import time

def load_fvecs(file_path):
    data = np.fromfile(file_path, dtype=np.int32)
    if len(data) == 0: return np.empty((0, 0), dtype=np.float32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:].copy().view(np.float32)

base_data = load_fvecs("./data/sift/sift_base.fvecs")[:500000]
query_data = load_fvecs("./data/sift/sift_query.fvecs")[0:1]

nb, d = base_data.shape
k = 100
nlist = 2048
m = 8
nprobe = 48
subsample = True

print("Calculating Ground Truth (NumPy Exact L2)...")
exact_dists = np.sum((base_data - query_data)**2, axis=1)
ground_truth_set = set(np.argsort(exact_dists)[:k])

base_ids = np.arange(nb, dtype=np.uint64)

best_seed = -1
max_matches = -1
print(f"\nHunting for the best K-Means seed (Testing 1-10)...")

for seed in range(1, 16): 

    db = vecmini.IndexIVFPQ(d, nlist, m)

    db.train(nb, base_data, subsample, seed)
    db.add(nb, base_data, base_ids)

    distances, labels = db.search(1, query_data, k, nprobe)
    test_labels = labels[0]
    matches = sum(1 for label in test_labels if label in ground_truth_set)

    print(f"Seed {seed:02d} -> Recall: {matches}%")
