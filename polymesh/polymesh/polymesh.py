import numpy as np
import xarray as xr
import spatialpandas as sp
import cartopy.crs as ccrs
import pygeos as pg
import pyarrow as pa





class Polymesh():
    def __init__(self, ugrid=None, projection=ccrs.PlateCarree()):
        """ Given a UXarray grid object, constructs a polygon
        mesh suitable for rendering with Datashader
        Parameters
        ----------
        ugrid : uxarray grid object, required
            Grid file name is the first argument.
        projection : ccrs., optional
            Cartopy projection for coordinate transform

        Returns
        -------
        object : Polymesh
            Class for creating renderable polygon meshes
        Examples
        --------
        Create a Poly Mesh object from a UXarray Dataset
        >>> grid_ds = ux.open_dataset(grid_filename)
        >>> mesh = Polymesh(grid_ds)
        """

        # Dictonary for Variables (remove on next UXarray Release)
        var_dict = ugrid.ds_var_names

        # Projection for Coordinate Transform
        self.projection = projection

        self.ds = ugrid.ds
        self.face_nodes = self.ds[var_dict['Mesh2_face_nodes']].values
        self.n_faces, self.n_face_nodes = self.face_nodes.shape

        # Face Node Index for Construction Polygons
        self.index = self.face_nodes.astype(int)

        # Original x and y coordinates
        x = self.ds[var_dict['Mesh2_node_x']].values
        y = self.ds[var_dict['Mesh2_node_y']].values

        # Calculate Index for Cyclic Cells based on Pre-Projection polygons
        self.drop_index = self.find_cyclic_polygons(x, y)

        # Create Polygon Array with Fixed Polygons
        self.polygon_array, self.new_poly_index = self.create_polygon_array(x, y)

        # Polygon Mesh Dataframe
        self.gdf = self.construct_mesh()



    def data_mesh(self, name, dims, fill='nodes'):
        """ Given a Variable Name and Dimensions, returns a
        GeoDataFrame containing geometry and fill values for
        the polygon mesh
        Parameters
        ----------
        name : string, required
            Name of data variable for rendering
        dims : dict, required
            Dictonary of dimensions for data variable
        fill : string
            Method for calculating face values

        Returns
        -------
        gdf : GeoDataFrame
            Contains polygon geometry and face values
        """

        # Ensure a valid variable name is passed through
        if name not in list(self.ds.data_vars):
            # add exception later
            print("Invalid Data Variable")
            return


        # Data is given for every 'face'
        if fill == 'faces':
            face_array = self.ds[name].isel(dims).values

        # Data is given for every 'face node'
        elif fill == 'nodes':
            face_array = np.zeros((self.polygon_array.shape[0]))
            # Face Values for Original Polygons
            face_array[:self.n_faces] = self.ds[name].isel(dims).values[self.index].mean(axis=1)

            # Face Values for New (Left & Right) Polygons
            if self.new_poly_index is not None:
                face_array[self.n_faces:] = face_array[self.new_poly_index]

        else:
            self.face_array = None

        # Face Values Excluding Cyclic Cells
        if self.new_poly_index is not None:
            updated_faces = np.delete(face_array, self.drop_index, axis=0)
            self.gdf = self.gdf.assign(faces = updated_faces)
        else:
            self.gdf = self.gdf.assign(faces = face_array)

        return self.gdf




    def construct_mesh(self):
        """ Constructs a Polygon Mesh using the calculated
        polygon array and drop index for cyclic polygons
        Parameters (from class)
        ----------
        polygon_array : ndarry
            Array containing Polygon Coordinates (original and new)
        drop_index : ndarray
            Array containing indices to cyclic polygons

        Returns
        -------
        gdf : GeoDataFrame
            Contains polygon geometry
        """

        # Create PyGeos Polygon Object
        if self.new_poly_index is not None:
            geo = pg.polygons(np.delete(self.polygon_array, self.drop_index, axis=0))
        else:
            geo = pg.polygons(self.polygon_array)

        # Get Coords and indicies for PyArrow
        arr_flat, part_indices = pg.get_parts(geo, return_index=True)
        offsets1 = np.insert(np.bincount(part_indices).cumsum(), 0, 0)
        arr_flat2, ring_indices = pg.geometry.get_rings(arr_flat, return_index=True)
        offsets2 = np.insert(np.bincount(ring_indices).cumsum(), 0, 0)
        coords, indices = pg.get_coordinates(arr_flat2, return_index=True)
        offsets3 = np.insert(np.bincount(indices).cumsum(), 0, 0)
        coords_flat = coords.ravel()
        offsets3 *= 2

        # Create a PyArrow array with our Polygons
        _parr3 = pa.ListArray.from_arrays(pa.array(offsets3), pa.array(coords_flat))
        _parr2 = pa.ListArray.from_arrays(pa.array(offsets2), _parr3)
        parr = pa.ListArray.from_arrays(pa.array(offsets1), _parr2)

        # Create Spatial Pandas Polygon Objects from PyArrow
        polygons = sp.geometry.MultiPolygonArray(parr)

        # Store our Polygon Geometry in a GeoDataFrame
        self.gdf = sp.GeoDataFrame({'geometry': polygons})

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PyGeos + PGPD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # x_coords = self.x[self.index]
        # y_coords = self.y[self.index]
        # self.polygon_array[:, :, 0] = x_coords
        # self.polygon_array[:, :, 1] = y_coords

        # polygons = pg.polygons(self.polygon_array)

        # df = pd.DataFrame({"geometry": polygons, "faces" : np.zeros(self.n_faces)})
        # df = df.astype({'geometry':'geos'})

        # self.df = df.geos.to_geopandas(geometry='geometry')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return self.gdf


    def find_cyclic_polygons(self, x, y):
        """ Finds cyclic polygons (longitude edges) and returns
        their indicies
        Parameters (from class)
        ----------
        x : ndarry
            coordinate values for 'x' coordinates
        y : ndarray
            coordinate values for 'y' coordinates

        Returns
        -------
        drop_index : ndarray
            Array containing indices to cyclic polygons
        """

        # Get Polygon Coordinate Data
        poly_data = np.zeros((self.n_faces, 2*self.n_face_nodes))
        poly_data[:, 0::2] = x[self.index]
        poly_data[:, 1::2] = y[self.index]

        # Find any polygon that has (+) & (-) x values
        xs = poly_data[:, 0::2]
        x_abs = np.abs(xs)
        out_left = np.all((-xs - x_abs) == 0, axis=1)
        out_right = np.all((xs - x_abs) == 0, axis=1)
        out = out_left | out_right

        # Store Index of Cyclic Cells
        drop_index = np.arange(0, self.n_faces, 1)
        drop_index = drop_index[~out]

        # Find all polygons between some (+) and (-) buffer
        center_buffer = 80
        poly_to_fix = poly_data[drop_index]
        corrected_index = np.any(np.abs(poly_to_fix[:, ::2]) < center_buffer, axis=1)

        # Update Cyclic Cells to exclude those center polygons
        drop_index = drop_index[~corrected_index]

        return drop_index



    def create_polygon_array(self, x, y):
        """ Converts coordinate and face node data to
        a polygon array, taking into account cyclic
        polygons
        Parameters (from class)
        ----------
        x : ndarry
            coordinate values for 'x' coordinates
        y : ndarray
            coordinate values for 'y' coordinates

        Returns
        -------
        polygon_array : ndarray
            Array containing Polygon Coordinates (original and new)
        """

        # Get Polygon Coordinate Data
        poly_data = np.zeros((self.n_faces, 2*self.n_face_nodes))
        poly_data[:, 0::2] = x[self.index]
        poly_data[:, 1::2] = y[self.index]

        # No Cyclic Polygons
        if len(self.drop_index) == 0:
            polygon_array = poly_data.reshape(self.n_faces, self.n_face_nodes, 2)
            return polygon_array, None

        # Get Cyclic Polygons
        poly_to_fix = poly_data[self.drop_index]

        # Itterate over each Cyclic Polygon, splitting it into two
        poly_list = []
        new_poly_index = []
        for i, poly in enumerate(poly_to_fix):
            poly_left = poly.copy()
            poly_right = poly.copy()

            # Start in RHS
            if poly[0] > 0:
                # Get Remaining x coordinates
                x_remain_index = poly[2::2] > 0

                # Update coordinates of Right Polygon
                poly_right[2::2][~x_remain_index] = poly[2::2][~x_remain_index] + 360

                # Update coordinates of Left Polygon
                poly_left[0] = poly[0] - 360
                poly_left[2::2][x_remain_index] = poly[2::2][x_remain_index] - 360

            # Start in LHS
            elif poly[0] < 0:
                # Get Remaining x coordinates
                x_remain_index = poly[2::2] < 0

                # Update coordinates of Left Polygon
                poly_left[2::2][~x_remain_index] = poly[2::2][~x_remain_index] - 360

                # Update coordinates of Right Polygon
                poly_right[0] = poly[0] + 360
                poly_right[2::2][x_remain_index] = poly[2::2][x_remain_index] + 360

            # Ignore
            else:
                # longitude = 0, might be the pole issue
                print("Missing Polygon at Index: {}".format(i))
                continue

            # Ensure longitude values are within +- 180 bound
            poly_left[::2][poly_left[::2] < -180] = -180.0
            poly_right[::2][poly_right[::2] > 180] = 180.0

            # Store New Polygons and Corresponding Indicies
            poly_list.extend([poly_left, poly_right])
            new_poly_index.extend([self.drop_index[i], self.drop_index[i]])


        n_new_faces = len(poly_list)
        new_poly_data = np.array(poly_list)
        new_poly_index = np.array(new_poly_index)

        # Number of Total Polygons (Original and New)
        n_total_polygons = self.n_faces + n_new_faces

        # Create a Polygon Array with new Left and Right Polygons
        polygon_array = np.zeros((n_total_polygons, self.n_face_nodes, 2))

        # Set Polygon Coordinates (With Transform)
        x_orig, y_orig, _ = self.projection.transform_points(ccrs.PlateCarree(),
                                                            poly_data[:, 0::2],
                                                            poly_data[:, 1::2]).T
        polygon_array[:self.n_faces, :, 0] = x_orig.T.astype(np.float32)
        polygon_array[:self.n_faces, :, 1] = y_orig.T.astype(np.float32)

        # Set Coordinates for new (left & right) polygons
        x_new, y_new, _ = self.projection.transform_points(ccrs.PlateCarree(),
                                                            new_poly_data[:, 0::2],
                                                            new_poly_data[:, 1::2]).T
        polygon_array[self.n_faces:, :, 0] = x_new.T
        polygon_array[self.n_faces:, :, 1] = y_new.T

        # Set Polygon Coordinates (No Transform)
        # polygon_array[:self.n_faces, :, 0] = poly_data[:, 0::2]
        # polygon_array[:self.n_faces, :, 1] = poly_data[:, 1::2]
        # polygon_array[self.n_faces:, :, 0] = new_poly_data[:, 0::2]
        # polygon_array[self.n_faces:, :, 1] = new_poly_data[:, 1::2]

        return polygon_array, new_poly_index




