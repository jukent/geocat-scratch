import numpy as np
import xarray as xr


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


def close_polygons():
    # add an edge going from the final vertex to the start vertex
    pass


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
