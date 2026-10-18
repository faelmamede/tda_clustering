import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from border_detect import BorderDetect
from cluster_calculator import ClusterCalculator
from dataset_generator_2d import DatasetGenerator2D
from dataset_generator_3d import DatasetGenerator3D
from tda_calculator import TDACalculator
import my_constants

class TopologicalDataAnalysisApplication:

    def plot_points_and_persistence_diagram(self, points, persistence, axes, show_grid=None, persistence_diagram_x_lim=None, persistence_diagram_y_lim=None):
        plt.subplots_adjust(wspace=0.3)
        ax_pts, ax_pd = axes

        self.plot_points(points, ax_pts, show_grid)
        self.plot_persistence_diagram(persistence, ax_pd, persistence_diagram_x_lim, persistence_diagram_y_lim)

    def plot_points(self, points, axes, show_grid=False):
        if self.DATASET_DIMENSION == '2D':
            points = np.array(points)
            axes.set_xlim(0, 1)
            axes.set_ylim(0, 1)
            axes.scatter(points[:,0], points[:,1], color='black', s=0.5)
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
        else:
            points = np.array(points)
            axes.scatter(points[:,0], points[:,1], points[:,2], color='black', s=0.5)
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')

    def plot_inc_pts(self, points, axes):
        if self.DATASET_DIMENSION == '2D':
            axes.scatter(points[:,0], points[:,1], color='blue', s=0.3)
        else:
            axes.scatter(points[:,0], points[:,1], points[:,2], color='blue', s=0.3)

    def plot_persistence_diagram(self, persistence, axes, x_lim=None, y_lim=None):
        data = [(p[0], p[1][0], p[1][1]) for p in persistence]
        arr = np.array(data)

        dims = arr[:, 0]
        births = arr[:, 1]
        deaths = arr[:, 2]

        inf_mask = (deaths == np.inf)
        finite_deaths = deaths[~inf_mask]
        max_val = np.max(finite_deaths) * 1.1 if finite_deaths.size > 0 else 1.0
        inf_y = max_val * 1.02
        deaths_plot = np.where(inf_mask, inf_y, deaths)

        config = {
            0: {'color': 'red', 'label': 'H0'},
            1: {'color': 'blue', 'label': 'H1'},
            2: {'color': 'green', 'label': 'H2'}
        }

        for d in sorted(np.unique(dims)):
            if d in config:
                mask = (dims == d)

                finite = mask & ~inf_mask
                if finite.any():
                    axes.scatter(births[finite], deaths_plot[finite],
                            color=config[d]['color'],
                            label=config[d]['label'],
                            alpha=0.6, s=30, edgecolors='none')

                inf_pts = mask & inf_mask
                if inf_pts.any():
                    label = config[d]['label'] if not finite.any() else '_nolegend_'
                    axes.scatter(births[inf_pts], deaths_plot[inf_pts],
                            color=config[d]['color'],
                            label=label,
                            alpha=0.6, s=30, edgecolors='none', marker='D')

        if inf_mask.any():
            lim_max_line = max_val * 1.05
            axes.axhline(y=inf_y, color='gray', linestyle='--', linewidth=1, alpha=0.8, label='∞')
            axes.text(lim_max_line * 0.01, inf_y, '∞', va='bottom', ha='left',
                    fontsize=11, color='gray')

        lims = [0, max_val * 1.05]
        axes.plot(lims, lims, 'k--', alpha=0.6, lw=1)

        if (x_lim is None) and (y_lim is None):
            x_lim = lims
            y_lim = [lims[0], inf_y * 1.03]

        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)
        axes.set_xlabel('Birth')
        axes.set_ylabel('Death')
        axes.legend(loc='lower right')
        axes.grid(True, linestyle=':', alpha=0.5)

        return x_lim, y_lim

    def plot_clusters(self, topological_cluster, dbscan_cluster, axes):
        ax_cluster_custom, ax_cluster_dbscan = axes

        unique_toplogical_clusters = sorted(topological_cluster.keys())
        colors = [plt.cm.hsv(i / len(unique_toplogical_clusters)) for i in range(len(unique_toplogical_clusters))]

        for i, cl in enumerate(unique_toplogical_clusters):
            pts = topological_cluster[cl]

            if self.DATASET_DIMENSION == '2D':
                ax_cluster_custom.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=0.5)
            else:
                ax_cluster_custom.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], s=0.5)

        ax_cluster_custom.set_aspect('equal')
        
        unique_dbscan_clusters = sorted(dbscan_cluster.keys())
        colors = [plt.cm.hsv(i / len(unique_dbscan_clusters)) for i in range(len(unique_dbscan_clusters))]

        for i, cl in enumerate(unique_dbscan_clusters):
            pts = dbscan_cluster[cl]

            if self.DATASET_DIMENSION == '2D':
                ax_cluster_dbscan.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=0.5)
            else:
                ax_cluster_dbscan.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], s=0.5)

        ax_cluster_dbscan.set_aspect('equal')

    def get_params_from_ui_fields(self):
        self.DATASET_DIMENSION = self.var_dataset_dimension.get()
        self.GRAPH_TYPE = self.var_graph_type.get()
        self.N_RAMDOM_SAMPLES = self.var_n_ramdom_samples.get()

        self.GRID_SIZE = int(self.var_grid_size.get())
        self.MAX_EDGE_LENGTH = float(self.var_max_edge_length.get())
        self.SPARSE = float(self.var_sparse.get())
        self.SELECTED_CLUSTER = self.var_selected_cluster.get()
        self.MORSE_FILTER = float(self.var_morse_filter.get())
        self.SEARCH_RADIUS = float(self.var_search_radius.get())

        print(f"DATASET DIMENSION: {self.DATASET_DIMENSION} | GRAPH: {self.GRAPH_TYPE}")
        print(f"N_RAMDOM_SAMPLES: {self.N_RAMDOM_SAMPLES}")

        print(f"GRID_SIZE: {self.GRID_SIZE}")
        print(f"SELECTED_CLUSTER: {self.SELECTED_CLUSTER}")
        print(f"MORSE_FILTER: {self.MORSE_FILTER}")
        print(f"SEARCH_RADIUS: {self.SEARCH_RADIUS}")

        print(f"MAX_EDGE_LENGTH: {self.MAX_EDGE_LENGTH}")
        print(f"SPARSE: {self.SPARSE}")
    
    def handle_dataset_generation(self):
        print("\n#################################################")
        print("\nGENERATING NEW DATASET... \n")

        # Limpar gráfico anterior
        ax_pts, ax_pd = self.axes_dataset
        ax_pts.clear()
        ax_pd.clear()

        self.get_params_from_ui_fields()

        datasetGenerator2D = DatasetGenerator2D(N_POINTS_RANDOM_SAMPLE=self.N_RAMDOM_SAMPLES)
        datasetGenerator3D = DatasetGenerator3D(N_POINTS_RANDOM_SAMPLE=self.N_RAMDOM_SAMPLES)

        if self.DATASET_DIMENSION == '2D':
            self.dataset = datasetGenerator2D.generate_dataset(self.GRAPH_TYPE)
        else:
            self.dataset = datasetGenerator3D.generate_dataset(self.GRAPH_TYPE)

        self.plot_points(self.dataset, ax_pts)

        tdaCalculator = TDACalculator(self.dataset)
        persistence, _, _ = tdaCalculator.apply()
        
        self.x_lim, self.y_lim = self.plot_persistence_diagram(persistence, ax_pd)

        self.plot_dataset_canvas.draw()
    
    def handle_find_borders(self):
        print("\n#################################################")
        print("\nFINDING TOPOLOGICAL BORDERS...\n")

        ax_pts, ax_pd = self.axes_tda_border
        ax_pts.clear()
        ax_pd.clear()

        self.get_params_from_ui_fields()

        borderDetect = BorderDetect(
            data=self.dataset,
            grid_size=self.GRID_SIZE,
            morse_filter=self.MORSE_FILTER,
            search_radius=self.SEARCH_RADIUS,
            is3D=True if self.DATASET_DIMENSION == '3D' else False
        )
  
        topological_borders = borderDetect.get_borders()

        if len(topological_borders) == 0:
            messagebox.showerror("No border was found", "No border was found! Change the parameters and try again!")
            return
        
        self.var_found_clusters.set(f"N Clusters: {len(topological_borders)}")

        borders_to_plot = []

        if self.SELECTED_CLUSTER == '0..n':
            for id in range(len(topological_borders)):
                pts_data = topological_borders[int(id)]['pts_data']
                pts_inc = topological_borders[int(id)]['pts_inc']

                self.plot_inc_pts(pts_inc, ax_pts)
                borders_to_plot.extend(pts_data)

        else:
            ids = self.SELECTED_CLUSTER.split(',')
            for id in ids:
                if int(id) >= len(topological_borders):
                    messagebox.showerror("Invalid cluster", f"The {id} cluster do not exist.")
                    return

                pts_data = topological_borders[int(id)]['pts_data']
                pts_inc = topological_borders[int(id)]['pts_inc']

                if len(pts_data) == 0:
                    messagebox.showerror("Invalid cluster", f"The {id} cluster {id} only has low-coherence cells.")

                self.plot_inc_pts(pts_inc, ax_pts)
                borders_to_plot.extend(pts_data)
        
        print("\nTOPOLOGICAL CLUSTERS SELECTED, PREPARE TO RUN TDA...\n")
        tdaCalculator = TDACalculator(borders_to_plot)
        persistence, _, _ = tdaCalculator.apply()
        self.plot_points_and_persistence_diagram(borders_to_plot, persistence, self.axes_tda_border, True, self.x_lim, self.y_lim)
        self.plot_borders_canvas.draw()

        ax_custom_cluster, ax_dbscan_cluster = self.axes_cluster_compare
        ax_custom_cluster.clear()
        ax_dbscan_cluster.clear()
        
        topological_clusters = {}
        for id in range(len(topological_borders)):
            topological_clusters[id] = topological_borders[int(id)]['pts_data']
        
        dbscan_cluster = {}
        clusterCalculator = ClusterCalculator()
        cluster_labels, _, _ = clusterCalculator.cluster(pts=self.dataset)
        
        unique_dbscan_clusters = sorted([l for l in set(cluster_labels) if l >= 0])

        for cl in unique_dbscan_clusters:
            pts_cluster = self.dataset[cluster_labels == cl]
            dbscan_cluster[cl] = pts_cluster

        self.plot_clusters(topological_cluster=topological_clusters, dbscan_cluster=dbscan_cluster, axes=self.axes_cluster_compare)
        self.plot_cluster_compare_canvas.draw()

    def change_dataset_dimension(self, event):
        self.DATASET_DIMENSION = self.var_dataset_dimension.get()  
  
        if self.DATASET_DIMENSION == '2D':  
            self.GRID_SIZE = 101  
            self.MORSE_FILTER = 0.1
            self.SEARCH_RADIUS = 5
            
            self.graph_type_box.config(values=my_constants.DATASET_2D_LIST)  
            self.var_graph_type.set(my_constants.DATASET_2D_LIST[0])  
        else:  
            self.GRID_SIZE = 41 
            self.MORSE_FILTER = 0.2
            self.SEARCH_RADIUS = 10
            
            self.graph_type_box.config(values=my_constants.DATASET_3D_LIST)  
            self.var_graph_type.set(my_constants.DATASET_3D_LIST[0])
        
        self.var_grid_size.set(str(self.GRID_SIZE))
        self.var_morse_filter.set(str(self.MORSE_FILTER))
        self.var_search_radius.set(str(self.SEARCH_RADIUS))

        self.reload_dataset_canvas()
        self.reload_borders_canvas()
        self.reload_cluster_compare_canvas()

    def set_control_frame(self):
        self.control_frame = ttk.Frame(self.content_frame, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.PAD = dict(padx=6, pady=4)

        dataset_frame = ttk.LabelFrame(self.control_frame, text="Dataset Params", padding="6")
        dataset_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 4))

        ttk.Label(dataset_frame, text="DIMENSION:").pack(side=tk.LEFT, **self.PAD)
        self.var_dataset_dimension = tk.StringVar(value=my_constants.DEFAULT_DATASET_DIMENSION)
        dataset_dimension_box = ttk.Combobox(dataset_frame, textvariable=self.var_dataset_dimension,
                                            values=["2D", "3D"], width=5)
        dataset_dimension_box.bind("<<ComboboxSelected>>", self.change_dataset_dimension)
        dataset_dimension_box.pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(dataset_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self.graph_type_label = ttk.Label(dataset_frame, text="GRAPH TYPE:")
        self.graph_type_label.pack(side=tk.LEFT, **self.PAD)
        self.var_graph_type = tk.StringVar(value=my_constants.DEFAULT_GRAPH_TYPE)
        self.graph_type_box = ttk.Combobox(dataset_frame, textvariable=self.var_graph_type,
                                            values=my_constants.DATASET_2D_LIST, width=14)
        self.graph_type_box.pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(dataset_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Label(dataset_frame, text="N SAMPLES:").pack(side=tk.LEFT, **self.PAD)
        self.var_n_ramdom_samples = tk.StringVar(value=my_constants.DAFAULT_N_POINTS_RANDOM_SAMPLE)
        ttk.Entry(dataset_frame, textvariable=self.var_n_ramdom_samples, width=8).pack(side=tk.LEFT, **self.PAD)

        ttk.Button(dataset_frame, text="Generate Dataset", command=self.handle_dataset_generation,
                width=16).pack(side=tk.RIGHT, padx=8, pady=4)

        ###################################################
        # ROW 1: BORDERS PARAMS
        ###################################################
        borders_frame = ttk.LabelFrame(self.control_frame, text="Border Params", padding="6")
        borders_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 4))

        # GRID SIZE
        self.grid_label = ttk.Label(borders_frame, text="GRID SIZE:")
        self.grid_label.pack(side=tk.LEFT, **self.PAD)
        self.var_grid_size = tk.StringVar(value=my_constants.DEFAULT_GRID_SIZE)
        self.grid_entry = ttk.Entry(borders_frame, textvariable=self.var_grid_size, width=6)
        self.grid_entry.pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(borders_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # SELECTED CLUSTER
        ttk.Label(borders_frame, text="SELECTED CLUSTER:").pack(side=tk.LEFT, **self.PAD)
        self.var_selected_cluster = tk.StringVar(value=my_constants.DEFAULT_SELECTED_CLUSTER)
        ttk.Entry(borders_frame, textvariable=self.var_selected_cluster, width=6).pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(borders_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # MORSE FILTER
        self.morse_filter_label = ttk.Label(borders_frame, text="MORSE FILTER:")
        self.morse_filter_label.pack(side=tk.LEFT, **self.PAD)
        self.var_morse_filter = tk.StringVar(value=my_constants.DEFAULT_MORSE_FILTER)
        self.morse_filter_entry = ttk.Entry(borders_frame, textvariable=self.var_morse_filter, width=6)
        self.morse_filter_entry.pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(borders_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # SEARCH RADIUS
        self.search_radius_label = ttk.Label(borders_frame, text="SEARCH RADIUS:")
        self.search_radius_label.pack(side=tk.LEFT, **self.PAD)
        self.var_search_radius = tk.StringVar(value=my_constants.DEFAULT_SEARCH_RADIUS)
        self.search_radius_entry = ttk.Entry(borders_frame, textvariable=self.var_search_radius, width=6)
        self.search_radius_entry.pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(borders_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # CLUSTERS label
        self.var_found_clusters = tk.StringVar(value="Clusters")  
        self.found_clusteres_label = ttk.Label(borders_frame, textvariable=self.var_found_clusters, foreground="gray")  
        self.found_clusteres_label.pack(side=tk.LEFT, **self.PAD)

        ###################################################
        # ROW 2: TDA PARAMS
        ###################################################
        tda_frame = ttk.LabelFrame(self.control_frame, text="TDA Params", padding="6")
        tda_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 4))

        # MAX EDGE LENGTH
        ttk.Label(tda_frame, text="MAX EDGE LENGTH:").pack(side=tk.LEFT, **self.PAD)
        self.var_max_edge_length = tk.StringVar(value=my_constants.DEFAULT_MAX_EDGE_LENGTH)
        ttk.Entry(tda_frame, textvariable=self.var_max_edge_length, width=6).pack(side=tk.LEFT, **self.PAD)

        ttk.Separator(tda_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # SPARSE
        ttk.Label(tda_frame, text="SPARSE:").pack(side=tk.LEFT, **self.PAD)
        self.var_sparse = tk.StringVar(value=my_constants.DEFAULT_SPARSE)
        ttk.Entry(tda_frame, textvariable=self.var_sparse, width=6).pack(side=tk.LEFT, **self.PAD)

        ttk.Button(tda_frame, text="Border Analysis", command=self.handle_find_borders,
                width=16).pack(side=tk.RIGHT, padx=8, pady=4)

    def set_dataset_frame(self):
        self.plot_dataset_frame = ttk.Frame(self.content_frame)  
        self.plot_dataset_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  
       
        self.reload_dataset_canvas()

        if self.DATASET_DIMENSION == '2D':
            datasetGenerator = DatasetGenerator2D()
            self.dataset = datasetGenerator.generate_elipse_and_2_circles()
        else:
            datasetGenerator = DatasetGenerator3D()
            self.dataset = datasetGenerator.generate_toro()
    
    def reload_dataset_canvas(self): 
        if hasattr(self, 'plot_dataset_canvas'):  
            self.plot_dataset_canvas.get_tk_widget().destroy() 

        if hasattr(self, 'plot_dataset_toolbar'):  
            self.plot_dataset_toolbar.destroy()  
      
        fig = Figure(figsize=(10, 4), dpi=100)  
    
        if self.DATASET_DIMENSION == '2D':  
            self.axes_dataset = fig.subplots(1, 2)  
        else:  
            gs = gridspec.GridSpec(1, 2, height_ratios=[1])  
            self.axes_dataset = (  
                fig.add_subplot(gs[0, 0], projection='3d'),  
                fig.add_subplot(gs[0, 1])  
            )  
    
        self.plot_dataset_canvas = FigureCanvasTkAgg(fig, master=self.plot_dataset_frame)  
        self.plot_dataset_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_dataset_toolbar = NavigationToolbar2Tk(self.plot_dataset_canvas, self.plot_dataset_frame)  
        self.plot_dataset_toolbar.update()

    def set_border_frame(self):
        self.plot_border_frame = ttk.Frame(self.content_frame)  
        self.plot_border_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  
        
        self.reload_borders_canvas()

    def reload_borders_canvas(self):  
        if hasattr(self, 'plot_borders_canvas'):  
            self.plot_borders_canvas.get_tk_widget().destroy()

        if hasattr(self, 'plot_borders_toolbar'):  
            self.plot_borders_toolbar.destroy()  
    
        fig = Figure(figsize=(10, 4), dpi=100)  
    
        if self.DATASET_DIMENSION == '2D':  
            self.axes_tda_border = fig.subplots(1, 2)  
        else:  
            gs = gridspec.GridSpec(1, 2, height_ratios=[1])  
            self.axes_tda_border = (  
                fig.add_subplot(gs[0, 0], projection='3d'),  
                fig.add_subplot(gs[0, 1])  
            )  
    
        self.plot_borders_canvas = FigureCanvasTkAgg(fig, master=self.plot_border_frame)  
        self.plot_borders_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
  
        self.plot_borders_toolbar = NavigationToolbar2Tk(self.plot_borders_canvas, self.plot_border_frame)  
        self.plot_borders_toolbar.update()

    def set_cluster_compare_frame(self):
        self.plot_cluster_compare_frame = ttk.Frame(self.content_frame)  
        self.plot_cluster_compare_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  
         
        self.reload_cluster_compare_canvas()

    def reload_cluster_compare_canvas(self):  
        if hasattr(self, 'plot_cluster_compare_canvas'):  
            self.plot_cluster_compare_canvas.get_tk_widget().destroy()  

        if hasattr(self, 'plot_cluster_compare_toolbar'):  
            self.plot_cluster_compare_toolbar.destroy() 
    
        fig = Figure(figsize=(10, 4), dpi=100)  
    
        if self.DATASET_DIMENSION == '2D':  
            self.axes_cluster_compare = fig.subplots(1, 2)  
        else:  
            gs = gridspec.GridSpec(1, 2, height_ratios=[1])  
            self.axes_cluster_compare = (  
                fig.add_subplot(gs[0, 0], projection='3d'),  
                fig.add_subplot(gs[0, 1], projection='3d') 
            )  
    
        self.plot_cluster_compare_canvas = FigureCanvasTkAgg(fig, master=self.plot_cluster_compare_frame)  
        self.plot_cluster_compare_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_cluster_compare_toolbar = NavigationToolbar2Tk(self.plot_cluster_compare_canvas, self.plot_cluster_compare_frame)  
        self.plot_cluster_compare_toolbar.update()

    def __init__(self, root):
        self.root = root
        self.root.title("TDA ANALYSIS")
        self.root.state("zoomed")

        self.DATASET_DIMENSION = my_constants.DEFAULT_DATASET_DIMENSION
        
        container = tk.Frame(root)
        container.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.scroll_canvas.yview)

        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.content_frame = ttk.Frame(self.scroll_canvas)

        def _on_frame_configure(event):
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            self.scroll_canvas.itemconfig(self.content_window, width=event.width)
        
        def _on_mousewheel(event):
            self.scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.content_window = self.scroll_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )

        self.content_frame.bind("<Configure>", _on_frame_configure)
        self.scroll_canvas.bind("<Configure>", _on_canvas_configure)
        self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self.set_control_frame()
        self.set_dataset_frame()
        self.set_border_frame()
        self.set_cluster_compare_frame()
        
        self.handle_dataset_generation()
        self.handle_find_borders()