import numpy as np

import my_constants

class DatasetGenerator3D:
    def __init__(self, N_POINTS_RANDOM_SAMPLE=my_constants.DAFAULT_N_POINTS_RANDOM_SAMPLE):  
        self.N_POINTS_RANDOM_SAMPLE = int(N_POINTS_RANDOM_SAMPLE)
    
    def _normalize(self, points):
        p_min = points.min(axis=0)
        max_span = np.max(points.max(axis=0) - p_min)
        pts_norm = (points - p_min) / max_span

        return pts_norm

    def generate_toro(self):
        u = 2 * np.pi * np.random.rand(self.N_POINTS_RANDOM_SAMPLE)
        v = 2 * np.pi * np.random.rand(self.N_POINTS_RANDOM_SAMPLE)

        c = np.array([0.5, 0.5, 0.5])

        R  = 0.22
        r0 = 0.08

        ruido = 0.002

        x = c[0] + (R + r0 * np.cos(v)) * np.cos(u) + ruido * np.random.randn(self.N_POINTS_RANDOM_SAMPLE)
        y = c[1] + (R + r0 * np.cos(v)) * np.sin(u) + ruido * np.random.randn(self.N_POINTS_RANDOM_SAMPLE)
        z = c[2] + r0 * np.sin(v)                   + ruido * np.random.randn(self.N_POINTS_RANDOM_SAMPLE)

        data = np.column_stack([x, y, z])

        return data

    def generate_sierpinski_tetrahedron(self, burn_in=50, seed=0):   
        rng = np.random.default_rng(seed)  
    
        V = np.array([  
            [0.0, 0.0, 0.0],  
            [1.0, 0.0, 0.0],  
            [0.0, 1.0, 0.0],  
            [0.0, 0.0, 1.0],  
        ], dtype=float)  
    
        p = rng.random(3)  
    
        pts = np.empty((self.N_POINTS_RANDOM_SAMPLE, 3), dtype=float)  
    
        for _ in range(burn_in):  
            v = V[rng.integers(0, 4)]  
            p = 0.5 * (p + v)  
    
        for i in range(self.N_POINTS_RANDOM_SAMPLE):  
            v = V[rng.integers(0, 4)]  
            p = 0.5 * (p + v)  
            pts[i] = p  
     
        mn = pts.min(axis=0)  
        mx = pts.max(axis=0)  
        span = np.where(mx - mn == 0, 1.0, mx - mn)  
        pts01 = (pts - mn) / span  
    
        return pts01

    def get_mistery_dataset(self):
        return self._normalize(np.loadtxt('C:\\Users\\rsilv\\Desktop\\mestrado\\TDA\\scripts\\dataset\\mistery.csv', delimiter=',', skiprows=1))

    def generate_dataset(self, type):
        if (type == "Toro"):
            dataset = self.generate_toro()
        elif (type == "Mistery"):
            dataset = self.get_mistery_dataset()
        elif (type == "Tetraedro de Sierpinski"):
            dataset = self.generate_sierpinski_tetrahedron()
        return dataset