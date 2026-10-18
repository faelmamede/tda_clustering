import numpy as np

from sklearn.cluster import DBSCAN  
from sklearn.neighbors import NearestNeighbors

class ClusterCalculator:
    def _calibrate_eps(self, points, k=5):  
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)  
        distances, _ = nbrs.kneighbors(points)  
        knn_dists = np.sort(distances[:, k-1])[::-1]  
    
        # Detecta cotovelo: maior diferença de segunda derivada  
        d2 = np.diff(np.diff(knn_dists))  
        elbow_idx = np.argmax(np.abs(d2)) + 1  
        eps_opt = knn_dists[elbow_idx]  
        return eps_opt, knn_dists  
    
    def cluster(self, pts, k=5):  
        eps_opt, knn_dists = self._calibrate_eps(pts, k=k)  
        print(f"eps calibrated automatically: {eps_opt:.5f}")  
    
        db = DBSCAN(eps=eps_opt, min_samples=3).fit(pts)  
        labels = db.labels_  
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  
        print(f"N clusters found: {n_clusters}  |  Noise: {(labels == -1).sum()} pts")  
        return labels, eps_opt, knn_dists   