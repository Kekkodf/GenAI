import numpy as np
import numpy.random as npr
import os
from .utils import vocab

class Mechanism:

    def __init__(self):
        self.vocab = vocab.read_embeddings(model="glove", dim=300)
        self.emb_matrix = np.array(list(self.vocab.values()))
        self.idx2word = {i: w for i, w in enumerate(self.vocab.keys())}
        self.word2idx = {w: i for i, w in enumerate(self.vocab.keys())}