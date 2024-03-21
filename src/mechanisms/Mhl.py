import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
from .Mechanism import Mechanism

"""
    @inproceedings{xu-etal-2020-differentially,
    title = "A Differentially Private Text Perturbation Method Using Regularized Mahalanobis Metric",
    author = "Xu, Zekun and Aggarwal, Abhinav and Feyisetan, Oluwaseyi and Teissier, Nathanael",
    editor = "Feyisetan, Oluwaseyi and Ghanavati, Sepideh  and Malmasi, Shervin and Thaine, Patricia",
    booktitle = "Proceedings of the Second Workshop on Privacy in NLP",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.privatenlp-1.2",
    doi = "10.18653/v1/2020.privatenlp-1.2",
    pages = "7--17",
    abstract = "Balancing the privacy-utility tradeoff is a crucial requirement of many practical machine learning systems that deal with sensitive customer data. A popular approach for privacy- preserving text analysis is noise injection, in which text data is first mapped into a continuous embedding space, perturbed by sampling a spherical noise from an appropriate distribution, and then projected back to the discrete vocabulary space. While this allows the perturbation to admit the required metric differential privacy, often the utility of downstream tasks modeled on this perturbed data is low because the spherical noise does not account for the variability in the density around different words in the embedding space. In particular, words in a sparse region are likely unchanged even when the noise scale is large. In this paper, we propose a text perturbation mechanism based on a carefully designed regularized variant of the Mahalanobis metric to overcome this problem. For any given noise scale, this metric adds an elliptical noise to account for the covariance structure in the embedding space. This heterogeneity in the noise scale along different directions helps ensure that the words in the sparse region have sufficient likelihood of replacement without sacrificing the overall utility. We provide a text-perturbation algorithm based on this metric and formally prove its privacy guarantees. Additionally, we empirically show that our mechanism improves the privacy statistics to achieve the same level of utility as compared to the state-of-the-art Laplace mechanism.",
    }
    """

class Mhl(Mechanism):
    
    def __init__(self, kwargs):
        super().__init__()
        self.epsilon = float(kwargs['epsilon']) if 'epsilon' in kwargs else 1.0
        self.lam = float(kwargs['lambda']) if 'lambda' in kwargs else 1.0
        self.cov_mat = np.cov(self.emb_matrix.T, ddof=0)
        self.sigma = self.cov_mat/ np.mean(np.var(self.emb_matrix.T, axis=1))
        _, self.m = self.emb_matrix.shape
        self.sigma_loc = sqrtm(self.lam * self.sigma + (1 - self.lam) * np.eye(self.m))
    
    def noise_sampling(self):
        N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
        X = N / np.sqrt(np.sum(N ** 2))
        X = np.dot(self.sigma_loc, X)
        X = X / np.sqrt(np.sum(X ** 2))
        Y = npr.gamma(self.m, 1 / self.epsilon)
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
        k = 1
        closest = np.argpartition(distance, k, axis=1)[:, :k]
        for i in range(len(query)):
            final_qry.append(list(self.vocab.keys())[closest[i][0]])
        return " ".join(final_qry)