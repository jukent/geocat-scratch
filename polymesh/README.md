<img src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo.png" data-canonical-src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo.png" width="380"/><br>

-----------------

# Python tool for visualizing unstructured grids as polygon meshes

## Description

## Requirements

## Installation

## Usage

```python
from polymesh import polymesh
import uxarray
grid_path = $GRID_PATH
data_path = $DATA_PATH

# Load Grid and Data files with UXarray
ds_grid = ux.open_dataset(grid_path, data_path)

# Construct Polygon Mesh
projection = ccrs.PlateCarree()
mesh = polymesh(ds=ds_grid, projection=projection)

# GeoDataFrame for Visualization with Datashader (Values at Edge Nodes)
df = mesh.data_mesh(name="Example Var", dims={"time" : 0}, fill='nodes')

# GeoDataFrame for Visualization with Datashader (Values at Faces)
df = mesh.data_mesh(name="Example Var", dims={"time" : 0}, fill='faces')
