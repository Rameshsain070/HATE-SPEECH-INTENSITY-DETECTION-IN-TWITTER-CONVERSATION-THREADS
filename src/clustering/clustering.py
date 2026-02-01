import numpy as np

try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None


class FuzzyClusterer:
    """
    Lightweight wrapper around fuzzy clustering.

    This was used to group reply-chain latent representations
    into soft clusters (each sample can belong to multiple clusters).
    """

    def __init__(self, n_clusters=5, m=2.0, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.random_state = random_state
        self.centers_ = None

    def fit(self, X):
        """
        Fit fuzzy c-means on latent vectors.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        """
        if fuzz is None:
            raise ImportError("scikit-fuzzy is required for fuzzy clustering")

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T,
            c=self.n_clusters,
            m=self.m,
            error=1e-4,
            maxiter=1000,
            init=None,
            seed=self.random_state
        )

        self.centers_ = cntr
        return u.T  # membership matrix [n_samples, n_clusters]

    def get_centers(self):
        if self.centers_ is None:
            raise RuntimeError("Clusterer has not been fitted yet")
        return self.centers_