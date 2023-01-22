import numpy as np
import shapely


def ugrid_to_polygon_coords(
        Mesh2_node_x: np.ndarray,
        Mesh2_node_y: np.ndarray,
        Mesh2_face_nodes: np.ndarray,
        fill_value=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    to-do
    """

    # handle fill value
    face_nodes = Mesh2_face_nodes.astype(int)
    polygon_x = Mesh2_node_x[face_nodes]
    polygon_y = Mesh2_node_y[face_nodes]
    return polygon_x, polygon_y


def close_polygons(
                    face_nodes : np.ndarray,
) -> np.ndarray:
    """Pads the (x) and (y) vertices representing
    a polygon (face) to form a closed polygons
    -------
    polygon_x : np.ndarray
        (x) coordinate of each polygon vertex

    polygon_y : np.ndarray
        (y) coordinate of each polygon vertex

    Returns
    -------
    polygon_x_pad : np.ndarray
        polygon_x padded to form a closed polygon
    polygon_y_pad : np.ndarray
        polygon_y padded to form a closed polygon
    """
    face_nodes_pad = np.pad(face_nodes, (0, 1), 'wrap')[:-1]

    return face_nodes_pad



def minmax_Longitude_rad(v1, v2):
    """Quantitative method to find the minimum and maximum Longitude between in a great circle
    Parameters
    ----------
    v1: float array
        The first endpoint of the great circle arc [lon, lat] in degree east.
    v1: float array
        The second endpoint of the great circle arc [lon, lat] in degree east.
    Returns
    -------
    float array
        [lon_min, lon_max] in radian
    """
    # First reorder the two ends points based on the rule: the span of its longitude must less than 180 degree
    [start_lon, end_lon] = np.sort([v1[0], v2[0]])
    if end_lon - start_lon <= 180:
        return [start_lon, end_lon]
    else:
        # swap the start and end longitude
        temp_lon = start_lon
        start_lon = end_lon
        end_lon = temp_lon
    return [start_lon, end_lon]





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

def create_polygon_array(x, y):


    # Get Polygon Coordinate Data
    poly_data = np.zeros((self.n_faces, 2 * self.n_face_nodes))
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


def find_antimeridian_faces():
    # for each face, find any segment that intersects the antimeridian
    pass


def split_antimeridian_faces():
    # split any face that intersects the antimeridian
    pass


def mpl_polygons():
    pass


def holoviz_polygons():
    pass
