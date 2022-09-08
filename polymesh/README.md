<img src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo-02.png" data-canonical-src="https://github.com/NCAR/geocat-scratch/blob/main/polymesh/docs/logo-02.png" width="380"/><br>

-----------------

# Python tool for representing unstructured data as polygon meshes for visualization 

PolyMesh utilities UXarray and PyGEOS for loading and representing unstructured data as 2D polygon meshes for rendering with Datashader. Functionality is focussed around supporting typical atmospheric science workflows (handling data on a sphere, support for geographic projections, etc), however suggestions for features from other domains are welcome. 

## Dependencies

### Conda Environment
`conda install --name polymesh --file requirements.txt --channel conda-forge`

## Installation 

PolyMesh exists as a class located within a single python file (polymesh/polymesh.py). Since it's not a package, there are various ways of importing and using the code.

### Python File 
One approach is to copy the polymesh.py file into your project and import the class from the relative path. For example, if you project is structured like this repository (project code in /notebooks/ and scripts in /polymesh/), you can add this when importing
```Python
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/polymesh")

from polymesh import Polymesh
```
### Notebooks
When working with notebooks, you may find it easiest to simply copy the entire PolyMesh class into your notebook. When doing this, make sure to copy the imports too.




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
df.hvplot.polygons(rasterize=True,aggregator='mean', c='faces', cmap=cmap)
```

## References
* [UXarray](https://github.com/UXARRAY/uxarray)
* [PyGEOS](https://github.com/pygeos/pygeos)
* [HoloViz](https://github.com/holoviz)
* [Joris van den Bossche's Datashader Issue](https://github.com/holoviz/datashader/issues/1006)
* [Unstructured Grid Visualization](https://scitools-iris.readthedocs.io/en/latest/further_topics/ugrid/data_model.html)

## Future
* Shapely 2.0 Improvements
* Integration with UXarray or GeoCAT-viz


