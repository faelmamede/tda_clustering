import numpy as np

from scipy.spatial import KDTree

class PreProcessing:
    def __init__(self, is3D):
        self.is3D = is3D

    def build_grid_centers(self, m): 
        if self.is3D: 
            js  = np.arange(1, m + 1)  
            is_ = np.arange(1, m + 1) 
            ks =  np.arange(1, m + 1) 

            JJ, II, KK = np.meshgrid(js, is_, ks)  

            cx = (2 * JJ - 1) / (2 * m)  
            cy = (2 * (m - II) + 1) / (2 * m)  
            cz = (2 * (m - KK) + 1) / (2 * m)  

            return cx, cy, cz 
        
        else: 
            js  = np.arange(1, m + 1)  
            is_ = np.arange(1, m + 1)  

            JJ, II = np.meshgrid(js, is_)  

            cx = (2 * JJ - 1) / (2 * m)    
            cy = (2 * (m - II) + 1) / (2 * m)  
 
            return cx, cy 

    def data2grid(self, data, m):  
        if self.is3D:
            cx, cy, cz = self.build_grid_centers(m) 

            centers = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])  

            tree = KDTree(data) 
  
            dists, _ = tree.query(centers, k=1, p=np.inf)  
 
            return centers[dists <= 1 / (2 * m)]
        else:
            cx, cy = self.build_grid_centers(m) 

            centers = np.column_stack([cx.ravel(), cy.ravel()])  

            tree = KDTree(data) 

            dists, _ = tree.query(centers, k=1, p=np.inf)  
 
            return centers[dists <= 1 / (2 * m)]