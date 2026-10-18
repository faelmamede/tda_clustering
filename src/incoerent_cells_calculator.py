import numpy as np

from scipy.spatial import KDTree

from pre_processing import PreProcessing

class IncoerentCellsCalculator:
    def __init__(self, data, grid_size, morse_filter, is3D=False):
        self.data = data
        self.grid_size = grid_size
        self.morse_filter = morse_filter
        self.is3D = is3D
        self.preProcessing = PreProcessing(is3D)
    
    def _morse_func(self, data_grid, grid_size):  
        m = grid_size

        if self.is3D:
            cx, cy, cz = self.preProcessing.build_grid_centers(m)  
            centers = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])  

            tree = KDTree(data_grid)  
 
            dists, _ = tree.query(centers, k=1)  

            return (dists ** 2).reshape(m, m, m)  
        else: 
            cx, cy = self.preProcessing.build_grid_centers(m)  
            centers = np.column_stack([cx.ravel(), cy.ravel()])  
  
            tree = KDTree(data_grid)

            dists, _ = tree.query(centers, k=1)  
  
            return (dists ** 2).reshape(m, m)  
    
    def _gradient(self, data, grid_size):
        m = grid_size

        if self.is3D:
            Dx = np.zeros((m, m, m))  
            Dy = np.zeros((m, m, m))  
            Dz = np.zeros((m, m, m))  
        
            Dy[1:-1, 1:-1, 1:-1] = (  
                m * (data[2:, 1:-1, 1:-1] - data[:-2, 1:-1, 1:-1]) / 2  
            )  

            Dx[1:-1, 1:-1, 1:-1] = (  
                m * (data[1:-1, 2:, 1:-1] - data[1:-1, :-2, 1:-1]) / 2  
            )  
         
            Dz[1:-1, 1:-1, 1:-1] = (  
                m * (data[1:-1, 1:-1, 2:] - data[1:-1, 1:-1, :-2]) / 2  
            )  
        
            return Dx, Dy, Dz  
        else:
            Dx = np.zeros((m, m))
            Dy = np.zeros((m, m))

            Dy[1:-1, 1:-1] = m * (data[2:, 1:-1] - data[:-2, 1:-1]) / 2  
            Dx[1:-1, 1:-1] = m * (data[1:-1, 2:] - data[1:-1, :-2]) / 2  
  
            return Dx, Dy  
    
    def _gradient_coherence(self, Dx, Dy, Dz=None, grid_size=None):  
        m = grid_size
        eps0 = 1e-12 

        if self.is3D:
            sl  = slice(1, m - 2)
            sl_plus  = slice(2, m - 1)  
            sl_minus = slice(0, m - 3)
        
            gx = np.stack([  
                Dx[sl,      sl,      sl],       
                Dx[sl_plus, sl,      sl],       
                Dx[sl_minus, sl,     sl],       
                Dx[sl,      sl_plus, sl],       
                Dx[sl,      sl_minus, sl],     
                Dx[sl,      sl,      sl_plus],
                Dx[sl,      sl,      sl_minus],
            ])
        
            gy = np.stack([  
                Dy[sl,      sl,      sl],  
                Dy[sl_plus, sl,      sl],  
                Dy[sl_minus, sl,     sl],  
                Dy[sl,      sl_plus, sl],  
                Dy[sl,      sl_minus, sl],  
                Dy[sl,      sl,      sl_plus],  
                Dy[sl,      sl,      sl_minus],  
            ])  
        
            gz = np.stack([  
                Dz[sl,      sl,      sl],  
                Dz[sl_plus, sl,      sl],  
                Dz[sl_minus, sl,     sl],  
                Dz[sl,      sl_plus, sl],  
                Dz[sl,      sl_minus, sl],  
                Dz[sl,      sl,      sl_plus],  
                Dz[sl,      sl,      sl_minus],  
            ])  
        
            norms = np.sqrt(gx**2 + gy**2 + gz**2)  
        
            valid = norms.min(axis=0) > eps0  
            safe  = np.where(norms > eps0, norms, 1.0)  
        
            mx = (gx / safe).mean(axis=0)  
            my = (gy / safe).mean(axis=0)  
            mz = (gz / safe).mean(axis=0)  
        
            coh = np.sqrt(mx**2 + my**2 + mz**2)  
        
            cf = np.full((m, m, m), -1.0)  
            cf[sl, sl, sl] = np.where(valid, coh, -1.0)  
        
            return cf
        else: 
            sl = slice(1, m - 2)  

            gx = np.stack([Dx[sl, sl],    Dx[sl, 2:m-1], Dx[sl, 0:m-3],  
                        Dx[2:m-1, sl], Dx[0:m-3, sl]])  
            
            gy = np.stack([Dy[sl, sl],    Dy[sl, 2:m-1], Dy[sl, 0:m-3],  
                        Dy[2:m-1, sl], Dy[0:m-3, sl]])  
            
            norms = np.sqrt(gx**2 + gy**2)  
            valid = norms.min(axis=0) > eps0  
            safe = np.where(norms > eps0, norms, 1.0)  

            mx = (gx / safe).mean(axis=0)  
            my = (gy / safe).mean(axis=0)  

            coh = np.sqrt(mx**2 + my**2) 

            cf = np.zeros((m - 1, m - 1))  
            cf[sl, sl] = np.where(valid, coh, -1.0)

            return cf  
    
    def _find_incoherence(self, cf, morse, grid_size):  
        m = grid_size
        r = self.morse_filter / (2  * m)

        sl = slice(1, m - 2)  

        if self.is3D:  
            mask = (cf[sl, sl, sl] <= 0.9) & (morse[sl, sl, sl] > r)  

            ii, jj, kk = np.where(mask)  

            ii += 1  
            jj += 1  
            kk += 1

            cx = (2 * jj - 1) / (2 * m)  
            cy = (2 * (m - ii) + 1) / (2 * m)  
            cz = (2 * (m - kk) + 1) / (2 * m)  
            
            return np.column_stack([cx, cy, cz]) if len(cx) > 0 else np.empty((0, 2))
        
        else:  
            sl = slice(1, m - 2)  

            mask = (cf[sl, sl] <= 0.9) & (morse[sl, sl] > r)  

            ii, jj = np.where(mask)  

            ii += 1  
            jj += 1  

            cx = (2 * jj - 1) / (2 * m)  
            cy = (2 * (m - ii) + 1) / (2 * m)  
            
            return np.column_stack([cx, cy]) if len(cx) > 0 else np.empty((0, 2))
    
    def _remove_frame(self, points, grid_size):
        m = grid_size
        pts = np.array(points, dtype=float)  
      
        x_min, y_min = pts.min(axis=0)  
        x_max, y_max = pts.max(axis=0)  
    
        dx = (x_max - x_min) / m if x_max != x_min else 1.0  
        dy = (y_max - y_min) / m if y_max != y_min else 1.0  
    
        cols = np.minimum(((pts[:, 0] - x_min) / dx).astype(int), m - 1)  
        rows = np.minimum(((pts[:, 1] - y_min) / dy).astype(int), m - 1)  
    
        mask = (  
            (cols != 0) & (cols != m - 1) &  
            (rows != 0) & (rows != m - 1)  
        )  
    
        return pts[mask]
    
    def calculate(self):
        data_grid = self.preProcessing.data2grid(self.data, self.grid_size)

        morse = self._morse_func(data_grid, self.grid_size)

        if self.is3D:
            Dx, Dy, Dz = self._gradient(morse, self.grid_size)
            coherence_field = self._gradient_coherence(Dx=Dx, Dy=Dy, Dz=Dz, grid_size=self.grid_size)
        else:
            Dx, Dy = self._gradient(morse, self.grid_size)
            coherence_field = self._gradient_coherence(Dx=Dx, Dy=Dy, grid_size=self.grid_size)

        incoherence_cells = self._find_incoherence(coherence_field, morse, self.grid_size)

        if self.is3D:
            return incoherence_cells, data_grid
        return self._remove_frame(incoherence_cells, self.grid_size), data_grid