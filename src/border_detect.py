import numpy as np

from scipy.spatial import KDTree

from incoerent_cells_calculator import IncoerentCellsCalculator
from cluster_calculator import ClusterCalculator

class BorderDetect:
    def __init__(self, data, grid_size, morse_filter, search_radius, is3D=False):
        self.data = data
        self.grid_size = grid_size
        self.is3D = is3D
        self.search_radius = search_radius
        self.incoerentCellsCalculator = IncoerentCellsCalculator(data=data, grid_size=grid_size, morse_filter=morse_filter, is3D=is3D)
        self.clusterCalculator = ClusterCalculator()

    def _find_borders_query_ball_point(self, incoherence_cells): 
        print("\Clustering low-coherence cells...")  
        cluster_labels, eps_opt, knn_dists = self.clusterCalculator.cluster(incoherence_cells, k=5)  

        radius = self.search_radius / self.grid_size

        tree_data = KDTree(self.data)

        unique_clusters = sorted([l for l in set(cluster_labels) if l >= 0])
        print(f"Valid clusters: {unique_clusters}")

        borders = {}

        for cl in unique_clusters:
            pts_inc = incoherence_cells[cluster_labels == cl]

            neighbor_lists = tree_data.query_ball_point(pts_inc, r=radius)

            neighbor_idx = np.unique(np.concatenate(neighbor_lists).astype(int)) if any(len(nl) > 0 for nl in neighbor_lists) else np.array([], dtype=int)

            pts_data = self.data[neighbor_idx]

            print(f"\n  Cluster {cl}: {pts_inc.shape[0]} low-coherence cells | {pts_data.shape[0]} pts")

            borders[cl] = {
                'pts_data': pts_data,
                'pts_inc': pts_inc
            }

        return borders

    def get_borders(self):
        incoherence_cells, data_grid = self.incoerentCellsCalculator.calculate()                                                                                                                                                                          
        borders = self._find_borders_query_ball_point(incoherence_cells)

        return borders