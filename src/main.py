import ttkbootstrap as tb
from ttkbootstrap.constants import *

from tda_app import TopologicalDataAnalysisApplication

# Criar e executar aplicação
if __name__ == "__main__":
    root = tb.Window(themename="lumen")
    app = TopologicalDataAnalysisApplication(root)
    root.mainloop()