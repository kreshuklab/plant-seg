# Hierarchical Design of PlantSeg

PlantSeg is organized into three layers:

 1. Functionals (Python API): The foundational layer of PlantSeg, providing its core functionality. This layer can be accessed directly in Python scripts or Jupyter notebooks.
 2. Tasks: The intermediate layer of PlantSeg, which encapsulates the functionals to handle resource management and support distributed computing.
 3. Napari Widgets: The top layer of PlantSeg, which integrates tasks into user-friendly widgets for easy interaction within graphical interfaces.
