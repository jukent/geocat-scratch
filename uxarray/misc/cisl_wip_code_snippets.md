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
```

# MPAS Example
```Python
import xarray as xr
import uxarray as ux

# path to mpas grid file with connectivity
mpas_grid_path = "/path/to/mpas_grid.nc"

# xarray dataset with MPAS connectivity variables
mpas_grid_ds = xr.open_dataset(mpas_grid_path)

# create a grid object from the primal mesh (two options)
primal_grid = ux.Grid(mpas_grid_ds)
primal_grid = ux.Grid(mpas_grid_ds, use_dual=False)

# create a grid object from the dual mesh
dual_grid = ux.Grid(mpas_grid_ds, use_dual=True)
```

# Coordinates
```Python
    Mesh2_node_x
     unit:  "degree_east"
    Mesh2_node_y
     unit:  "degree_north"
    Mesh2_node_z
     unit:  "m"
    Mesh2_node_cart_x
     unit:  "m"
    Mesh2_node_cart_y
     unit:  "m"
    Mesh2_node_cart_z
     unit:  "m"
```
# Redesign Usage example

```Python
import uxarray as ux

# grid and data file paths
grid_path = "/path/to/grid.nc"
data_path = "/path/to/data.nc"

# uxarray dataset with grid accessor
ux_ds = ux.open_dataset(grid_path, data_path)

# acessing data variables (returns a ux.DataArray)
psi = ux_ds['PSI']

# accessing grid connectivity through the accessor
face_nodes = ux_ds.uxgrid['Mesh2_face_nodes']
face_nodes = ux_ds.uxgrid.Mesh2_face_nodes

# other examples
face_areas = ux_ds.uxgrid.compute_face_areas()
integral = ux_ds['PSI'].uxgrid.integrate()

```
# non-redesign example

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

# acessing data variables 
psi = data_ds['PSI']

# accessing grid connectivity variables
face_nodes = grid.ds['Mesh2_face_nodes']
face_nodes = grid.Mesh2_face_nodes

# other examples
face_areas = ux_ds.uxgrid.compute_face_areas()
integral = ux_ds['PSI'].uxgrid.integrate()


