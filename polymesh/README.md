<img src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo-02.png" data-canonical-src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo-02.png" width="380"/><br>

-----------------

# Python tool for visualizing unstructured grids as polygon meshes

## Description

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

# Visualization
df.hvplot.polygons(rasterize=True,aggregator='mean', c='faces', cmap=cmap) * gf.coastline(projection=projection) * gf.borders(projection=projection)
```

## References
* [UXarray](https://github.com/UXARRAY/uxarray)
* [PyGEOS](https://github.com/pygeos/pygeos)
* [HoloViz](https://github.com/holoviz)
* [Joris van den Bossche's Datashader Issue](https://github.com/holoviz/datashader/issues/1006)
