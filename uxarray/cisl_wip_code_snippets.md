# Usage Example for CISL WIP

```Python
import xarray as xr
import uxarray as ux

# grid and data file paths
grid_path = "/path/to/grid.nc"
data_path = "/path/to/data.nc"

# dataset with grid connectivity
grid_ds = xr.open_dataset(grid_path)

# dataset with data variables 
data_ds = xr.open_dataset(data_path)

# uxarray grid object constructed from xarray dataset
grid = ux.Grid(grid_ds)

# examples
x_coords = grid.Mesh2_node_x
face_nodes = grid.Mesh2_face_nodes
face_areas = grid.compute_face_areas()
integral = grid.integrate(data_ds['PSI'])
