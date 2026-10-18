import random
import numpy as np

import my_constants

class DatasetGenerator2D:
    def __init__(self, N_POINTS_RANDOM_SAMPLE=my_constants.DAFAULT_N_POINTS_RANDOM_SAMPLE):  
        self.N_POINTS_RANDOM_SAMPLE = int(N_POINTS_RANDOM_SAMPLE)

    def generate_rotated_ellipse(self, n, angle):
        center = np.array([0.25, 0.75])
        radius = 0.05

        x = 0.1 * np.cos(angle) + np.random.rand(n) / 50
        y = 0.05 * np.sin(angle) + np.random.rand(n) / 50

        rot_matrix = np.sqrt(2) * np.array([[1, -1], [1,  1]])

        XY = rot_matrix @ np.vstack([x, y])

        x = center[0] + XY[0, :]
        y = center[1] + XY[1, :]

        return x, y

    def generate_circle(self, center, radius, n, angle):
        x = center[0] + radius * np.cos(angle) + np.random.rand(n) / 50
        y = center[1] + radius * np.sin(angle) + np.random.rand(n) / 50

        return x, y

    def generate_elipse_and_2_circles(self):
        n = int(self.N_POINTS_RANDOM_SAMPLE / 3)

        angle = 2 * np.pi * np.random.rand(n)

        x1, y1 = self.generate_rotated_ellipse(n, angle)
        x2, y2 = self.generate_circle(np.array([0.75, 0.25]), 0.05, n, angle)
        x3, y3 = self.generate_circle(np.array([0.25, 0.25]), 0.05, n, angle)

        data = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
            np.column_stack([x3, y3])
        ])

        return data
    
    def generate_2_circles(self):
        n = int(self.N_POINTS_RANDOM_SAMPLE / 2)
        
        angle = 2 * np.pi * np.random.rand(n)

        x1, y1 = self.generate_circle(np.array([0.25, 0.25]), 0.05, n, angle)
        x2, y2 = self.generate_circle(np.array([0.40, 0.25]), 0.10, n, angle)

        data = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
        ])

        return data

    def generate_sierpinski_triangule(self):
            pts = np.array([
                (0, 0),
                ((my_constants.N_POINTS / 100), 0),
                ((my_constants.N_POINTS / 100) / 2, (my_constants.N_POINTS / 100) / 2)
            ], dtype=np.float32)

            pt = np.array((0, 0), dtype=np.float32)

            x_points = []
            y_points = []

            for i in range(my_constants.N_POINTS):
                s = random.randint(0, 2)
                pt = (pts[s] + pt) // 2
                x_points.append(pt[0])
                y_points.append(pt[1])
            
            data = np.column_stack([x_points, y_points])

            data_min = data.min(axis=0)
            data_max = data.max(axis=0)

            return self.sample_dataset((data - data_min) / (data_max - data_min))

    def generate_sierpinski_carpet(self):  
        pts = [  
            (0, 0), (0, 1), (0, 2),  
            (1, 0),         (1, 2),  
            (2, 0), (2, 1), (2, 2)  
        ]  
        x, y = 0, 0  
        x_points = []  
        y_points = []  
    
        for _ in range(my_constants.N_POINTS):  
            s = random.choice(pts)  
            x = (x + s[0]) / 3  
            y = (y + s[1]) / 3  
            x_points.append(x)  
            y_points.append(y)  

        return self.sample_dataset(np.column_stack([x_points, y_points]))
    
    def generate_circular_holes_dataset(  
        self,
        x_range=(0, 100),  
        y_range=(0, 100),  
        holes=None,  
        random_state=42  
    ):  
        rng = np.random.default_rng(random_state)  
        
        if holes is None:  
            holes = [  
                (0.3, 0.3, 0.1),
                (0.7, 0.4, 0.08),
                (0.5, 0.7, 0.12)
            ]  
         
        x = rng.uniform(0, 1, size=my_constants.N_POINTS * 3)  
        y = rng.uniform(0, 1, size=my_constants.N_POINTS * 3) 
        
        points = np.column_stack([x, y])  
         
        mask_keep = np.ones(len(points), dtype=bool)  
        
        for (cx, cy, r) in holes:  
            dx = points[:, 0] - cx  
            dy = points[:, 1] - cy  
            dist2 = dx**2 + dy**2  
            mask_keep &= (dist2 > r**2)  
        
        points = points[mask_keep]  
        
        if len(points) > my_constants.N_POINTS:  
            points = points[:my_constants.N_POINTS]  

        return self.sample_dataset(points)

    def generate_variable_density_swiss(self, random_state=42):
        rng = np.random.default_rng(random_state)

        raw_points = rng.standard_normal((my_constants.N_POINTS * 10, 2)) * 0.2 + 0.5
        
        points = raw_points[(raw_points[:,0] >= 0) & (raw_points[:,0] <= 1) & 
                            (raw_points[:,1] >= 0) & (raw_points[:,1] <= 1)]
        
        holes = [(0.5, 0.5, 0.1), (0.2, 0.8, 0.06), (0.8, 0.2, 0.06)]
        
        mask = np.ones(len(points), dtype=bool)
        for cx, cy, r in holes:
            dist = np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2)
            mask &= (dist > r)
        
        return self.sample_dataset(points[mask][:my_constants.N_POINTS])

    def generate_complex_c_shape(self, random_state=42):
        rng = np.random.default_rng(random_state)
        points = rng.uniform(0, 1, (my_constants.N_POINTS * 3, 2))
        
        def is_in_c_shape(x, y):
            cx, cy = 0.5, 0.5
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)

            is_ring = (dist > 0.12) & (dist < 0.28)

            is_not_mouth = ~((x > 0.5) & (y > 0.4) & (y < 0.6))
            return is_ring & is_not_mouth

        mask = ~is_in_c_shape(points[:,0], points[:,1])
        return self.sample_dataset(points[mask][:my_constants.N_POINTS])
    
    def generate_shared_hole(self, random_state=42):   
        rng = np.random.default_rng(random_state)  
        
        points = rng.uniform(0, 1, (my_constants.N_POINTS * 5, 2))  
        
        cx_left,  cy_left  = 0.3, 0.5  
        r_inner_left       = 0.08 
        r_outer_left       = 0.20 
         
        cx_right, cy_right = 0.7, 0.5  
        r_inner_right      = 0.08  
        r_outer_right      = 0.20  
        
        dist_left  = np.sqrt((points[:,0] - cx_left )**2 + (points[:,1] - cy_left )**2)  
        dist_right = np.sqrt((points[:,0] - cx_right)**2 + (points[:,1] - cy_right)**2)  
         
        in_left_ring  = (dist_left  >= r_inner_left ) & (dist_left  <= r_outer_left )  
        in_right_ring = (dist_right >= r_inner_right) & (dist_right <= r_outer_right)  
        
        mask = in_left_ring | in_right_ring  
        
        points = points[mask][:my_constants.N_POINTS]  
        
        metadata = {  
            'holes': [  
                {'cx': cx_left,  'cy': cy_left,  'r': r_inner_left },  
                {'cx': cx_right, 'cy': cy_right, 'r': r_inner_right},  
            ],  
            'rings': [  
                {'cx': cx_left,  'cy': cy_left,  'r_inner': r_inner_left,  'r_outer': r_outer_left },  
                {'cx': cx_right, 'cy': cy_right, 'r_inner': r_inner_right, 'r_outer': r_outer_right},  
            ]  
        }  
        
        return self.sample_dataset(points)
    
    def generate_concentric_rings(self, random_state=42):  
        rng = np.random.default_rng(random_state)  
        
        points = rng.uniform(0, 1, (my_constants.N_POINTS * 5, 2))  
          
        cx, cy = 0.5, 0.5  
        dist = np.sqrt((points[:,0] - cx)**2 + (points[:,1] - cy)**2)  
          
        solid_bands = [  
            (0.00, 0.05),  
            (0.12, 0.18),
            (0.25, 0.31),
            (0.38, 0.44),
        ]  
        
        hole_bands = [  
            (0.05, 0.12),
            (0.18, 0.25),
            (0.31, 0.38),
        ]  
        
        mask = np.zeros(len(points), dtype=bool)  
        for r_min, r_max in solid_bands:  
            mask |= (dist >= r_min) & (dist <= r_max)  
        
        points = points[mask][:my_constants.N_POINTS]  
        
        metadata = {  
            'center': (cx, cy),  
            'solid_bands': solid_bands,  
            'hole_bands': hole_bands,  
            'n_holes': len(hole_bands)  
        }  
        
        return self.sample_dataset(points)

    def generate_dataset(self, type):
        if (type == "2 circles, 1 elipse"):
            dataset = self.generate_elipse_and_2_circles()
        if (type == "2 circles"):
            dataset = self.generate_2_circles()
        elif (type == "Triangulo de Sierpinksi"):
            dataset = self.generate_sierpinski_triangule()
        elif (type == "Tapete de Sierpinksi"):
            dataset = self.generate_sierpinski_carpet()
        elif (type == "Buracos circulares"):
            dataset = self.generate_circular_holes_dataset()
        elif (type == "Queijo suiço"):
            dataset = self.generate_variable_density_swiss()
        elif (type == "Formato de C"):
            dataset = self.generate_complex_c_shape()
        elif (type == "Buraco Compartilhado"):
            dataset = self.generate_shared_hole()
        elif (type == "Anéis Concêntricos"):
            dataset = self.generate_concentric_rings()
        return dataset

    def sample_dataset(self, dataset):
        random_indexes = sorted(random.sample(range(0, len(dataset[:,0])), self.N_POINTS_RANDOM_SAMPLE))

        x_sampled = []
        y_sampled = []
        for index in random_indexes:
            x_sampled.append(dataset[:,0][index])
            y_sampled.append(dataset[:,1][index])

        return np.column_stack([x_sampled, y_sampled])