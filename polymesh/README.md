<img src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo-02.png" data-canonical-src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo-02.png" width="380"/><br>

-----------------

# Python tool for representing unstructrued grids as polygon meshes

## Unstructured Grids
[short background about unstructrued grids and UXarray]

<img src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/data_ugrid_mesh.svg" data-canonical-src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/data_ugrid_mesh.svg" width="700"/><br>

## Cyclic Polygons on a Sphere
When constructing a polygon mesh from data that exists on a sphere, there may exist grid cells that lie on the boundary between positive and negative 180 longitude. These polygons would be rendered as thin, long strips due to their difference in longitude coordinates. To address this, PolyMesh locates these cyclic polygons, splits them up into two mirrors of the original (left and right), clips them, and masks the original. This allows for us to visualize the flat projection of our data, without any the artifacts present with the original mesh.

## PyGEOS, SpatialPandas, and HoloViz
By using PyGEOS and SpatialPandas to handle mesh construction, it a fraction of the time that Delaunay Triangulation would require (~4x Faster), while also yeilding a direct reconstruction of our original unstructured grid (other than fixing the cyclic polygons). By rasterizing our plots with HoloViz (hvPlot, Datashader), it allows for the rendering of millions of polygons in less than a few seconds.

## Installation
When working directly within the /notebooks/ directory, we can use the code below to import the tool 
```Python
# Relative Path for PolyMesh
%load_ext autoreload
%autoreload 2

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/polymesh")

from polymesh import Polymesh as Polymesh
```

If you are working on your own project, you can simply copy the `polymesh.py` file into your project and import it from there. 

## Usage

```python
from polymesh import Polymesh
import uxarray as uxr
grid_path = $GRID_PATH
data_path = $DATA_PATH

# Load Grid and Data files with UXarray
ds_grid = uxr.open_dataset(grid_path, data_path)

# Construct Polygon Mesh
projection = ccrs.PlateCarree()
mesh = Polymesh(ds=ds_grid, projection=projection)
mesh.construct_mesh()

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
* [Unstructured Grid Visualization](https://scitools-iris.readthedocs.io/en/latest/further_topics/ugrid/data_model.html)

## Future
* Shapely 2.0 Improvements
