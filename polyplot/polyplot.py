import numpy as np
import xarray as xr
import pandas as pd
import dask as da
from dask import delayed, compute
import scipy.spatial
import spatialpandas as sp
import cartopy.crs as ccrs
import cmocean

from multiprocessing import Pool

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize
import datashader as ds
import geoviews.feature as gf





class poly_plot():

    def __init__(self, x, y, face_nodes, n_workers=1):
        """_summary_

        Args:   
            x (_type_): _description_
            y (_type_): _description_
            face_nodes (_type_): _description_
        """
        # UGRID Data 
        # self.x, self.y = x, y
        # self.face_nodes = face_nodes
        # self.n_faces, self.n_face_nodes = face_nodes.shape
        # self.n_mesh_nodes = x.shape[0]
        #self.index = face_nodes.astype(int)

        # UGRID Data (Dask)
        self.n_workers = n_workers
        self.x = da.array.from_array(x)
        self.y = da.array.from_array(y)
        self.face_nodes = da.array.from_array(face_nodes)
        self.index = da.array.from_array(face_nodes.astype(int))
        self.n_faces, self.n_face_nodes = face_nodes.shape
        self.n_mesh_nodes = x.shape[0]
    
        # Polygon Mesh Data (polygons and faces)
        self.df = None
        self.df_fixed = None
        self.face_array = None
        self.poly_array = np.zeros((self.n_faces, 2*self.n_face_nodes),
                                   dtype=np.float64)

    def set_data(self, data):
        '''Set Data Variable'''

        # Calculate face value based on # of face nodes
        self.face_array = np.mean(data[self.index], axis=1)

        # # Initial Data Setup
        # if self.face_array is None:
        #     self.face_array = face_array
        # else:
        #     self.df['faces'] = self.face_array.tolist()
        #     self.fix_cells() 
    
    def to_poly_mesh(self):
        '''Construct Polygon Mesh'''
        # Point Index
        p_index = np.arange(0, 2*self.n_face_nodes)

        # Vectorized Approach
        for n, i, j in zip(range(self.n_face_nodes), p_index[::2], p_index[1::2]):
            self.poly_array[:, i] = self.x[self.index[:, n]]
            self.poly_array[:, j] = self.y[self.index[:, n]]

        # Create Polygon and Face Dataframe
        self.poly_array = self.poly_array.reshape((self.n_faces, 1, 2*self.n_face_nodes))
        polygons = sp.geometry.PolygonArray(self.poly_array.tolist())

        if self.face_array is None:
            self.df = sp.GeoDataFrame({'geometry': polygons})
        else:
            self.df = sp.GeoDataFrame({'geometry': polygons,
                                       'faces': self.face_array.tolist()})

        # Correct edge cells
        self.fix_cells()

    def construct_polygons(self, n):
        x = self.x[self.index[n]]
        y = self.
        polygon_row = np.insert(x, np.arange(len(x)), y)
        return polygon_row

    def calculate_face(self, x, y):
        return

    
    def to_poly_mesh_parallel(self):
        
        index = self.face_nodes.astype(int)

        lazy_polygons = []
        for n in range(self.n_faces):
            polygon = delayed(self.construct_polygon)(self.x[index[n]],
                                                 self.y[index[n]])
            lazy_polygons.append(polygon)

        


            




    def fix_cells(self):
        '''Fixes cells near the edges of the map (+-180 lon)'''
        n_poly = self.df['geometry'].values.shape[0]
        poly_data = self.df['geometry'].values.buffer_values.reshape(n_poly, 2*self.n_face_nodes)
        face_data = self.df['faces'].values

        # Get x values
        x = poly_data[:, ::2]
        x_abs = np.abs(poly_data[:, ::2])

        # Find violater cells (+) and (-) lon vals
        out_left = np.all((-x - x_abs) == 0, axis=1)
        out_right = np.all((x - x_abs) == 0, axis=1)
        out = out_left | out_right

        # Index of violater cells
        drop_index = np.arange(0, n_poly, 1)
        drop_index = drop_index[~out]

        # Get cells to fix
        poly_to_fix = poly_data[drop_index]

        # Exclude any center cells (-100, 100) lon
        corrected_index = np.any(np.abs(poly_to_fix[:, ::2]) < 100, axis=1)
        poly_to_fix = poly_to_fix[~corrected_index]
        drop_index = drop_index[~corrected_index]

        poly_left_list = []
        poly_right_list = []
        face_list = []

        # Split each violater cell into two cells
        for i, poly in enumerate(poly_to_fix):
            poly_left = poly.copy()
            poly_right = poly.copy()
             
            # Start in RHS
            # [2::2] -> every x value excluding starting point
            if poly[0] > 0:
                x_remain_index = poly[2::2] > 0
                poly_right[2::2][~x_remain_index] = poly[2::2][~x_remain_index] + 360 
                
                poly_left[0] = poly[0] - 360
                poly_left[2::2][x_remain_index] = poly[2::2][x_remain_index] - 360

                poly_left_list.append(poly_left)
                poly_right_list.append(poly_right)
                face_list.append(face_data[i])
                continue
            # Start in LHS
            elif poly[0] < 0:
                x_remain_index = poly[2::2] < 0
                poly_left[2::2][~x_remain_index] = poly[2::2][~x_remain_index] - 360

                poly_right[0] = poly[0] + 360
                poly_right[2::2][x_remain_index] = poly[2::2][x_remain_index] + 360
                
                poly_left_list.append(poly_left)
                poly_right_list.append(poly_right)
                face_list.append(face_data[i])
                continue
            # Ignore
            else:
                continue

        # Drop Violater Cells
        df_droped = self.df.drop(drop_index)

        # Create polygon and face arrays to add to df
        face_insert = np.concatenate([face_list, face_list])
        poly_insert = np.concatenate([poly_left_list, poly_right_list])
        poly_insert = poly_insert.reshape(2*len(poly_left_list), 1, 2*self.n_face_nodes)

        # Create new df with left and right cells
        polygons = sp.geometry.PolygonArray(poly_insert.tolist())
        df_insert = sp.GeoDataFrame({'geometry': polygons,
                                    'faces': face_insert.tolist()})
    
        # Join existing and new df
        self.df_fixed = pd.concat([df_insert, df_droped], ignore_index=True)
        
        return self.df_fixed

    def mesh(self):
        '''Return Polygon Mesh For Plotting with Datashader'''
        return self.df_fixed
    
    def plot_data(self):
        '''not implemented'''
        return 
