import numpy as np
from tqdm import tqdm


embeddings = {}
path = f"./config/embeddings"
def read_embeddings(model="glove", dim=300):
    with open(path + f"/{model}.6B.{dim}d.txt", 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading {model} embeddings, dim={dim} [Default: GloVe, dim=300]"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings