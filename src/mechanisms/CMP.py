import numpy as np
import numpy.random as npr
from .Mechanism import Mechanism
"""
    @inproceedings{FeyisetanEtAl2020CMP,
    author = {Feyisetan, Oluwaseyi and Balle, Borja and Drake, Thomas and Diethe, Tom},
    title = {Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations},
    year = {2020},
    isbn = {9781450368223},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3336191.3371856},
    doi = {10.1145/3336191.3371856},
    abstract = {Accurately learning from user data while providing quantifiable privacy guarantees provides an opportunity to build better ML models while maintaining user trust. This paper presents a formal approach to carrying out privacy preserving text perturbation using the notion of d_χ-privacy designed to achieve geo-indistinguishability in location data. Our approach applies carefully calibrated noise to vector representation of words in a high dimension space as defined by word embedding models. We present a privacy proof that satisfies d_χ-privacy where the privacy parameter $varepsilon$ provides guarantees with respect to a distance metric defined by the word embedding space. We demonstrate how $varepsilon$ can be selected by analyzing plausible deniability statistics backed up by large scale analysis on GloVe and fastText embeddings. We conduct privacy audit experiments against $2$ baseline models and utility experiments on 3 datasets to demonstrate the tradeoff between privacy and utility for varying values of varepsilon on different task types. Our results demonstrate practical utility (< 2\% utility loss for training binary classifiers) while providing better privacy guarantees than baseline models.},
    booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
    pages = {178-186},
    numpages = {9},
    keywords = {privacy, plausible deniability, differential privacy},
    location = {Houston, TX, USA},
    series = {WSDM '20}
    }
    """

class CMP(Mechanism):
    
    def __init__(self, kwargs):
        super().__init__()
        self.epsilon = float(kwargs['epsilon']) if 'epsilon' in kwargs else 1.0

    #definition of the sampling noise function
    def noise_sampling(self):
        _, m = self.emb_matrix.shape
        N = self.epsilon * npr.multivariate_normal(np.zeros(m), np.eye(m))
        X = N / np.sqrt(np.sum(N ** 2))
        Y = npr.gamma(m, 1 / self.epsilon)
        Z = Y * X
        return Z
    
    def obfuscate(self, query):
        final_qry = []
        noisy_emb = []
        for word in query:
            if word in self.vocab:
                noisy_emb.append(np.array(self.vocab[word] + self.noise_sampling()))
            else:
                noisy_emb.append(np.zeros(300)+self.noise_sampling())   
        noisy_emb = np.array(noisy_emb)     
        def euclidean_distance_matrix(x, y):
            x_expanded = x[:, np.newaxis, :]
            y_expanded = y[np.newaxis, :, :]
            return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))
        distance = euclidean_distance_matrix(noisy_emb, self.emb_matrix)
        #use argapartition to find the k closest words
        k = 1
        closest = np.argpartition(distance, k, axis=1)[:, :k]
        for i in range(len(query)):
            final_qry.append(list(self.vocab.keys())[closest[i][0]])
        return "".join(final_qry)