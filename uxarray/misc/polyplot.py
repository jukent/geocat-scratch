import numpy as np
import xarray as xr
import pandas as pd
import dask as da
from dask import delayed, compute
import scipy.spatial
import spatialpandas as sp
import cartopy.crs as ccrs
import cmocean

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize
import datashader as ds
import geoviews.feature as gf





class poly_plot():
    def __init__(self, x, y, face_nodes, data=None):
        # UGRID Data 
        self.x, self.y = x, y 
        self.face_nodes = face_nodes
        self.index = face_nodes.astype(int)
        self.n_faces, self.n_face_nodes = face_nodes.shape
        self.n_mesh_nodes = x.shape[0]

        # Polygon Mesh Data (polygons and faces)
        self.df = None
        self.face_array = None if data is None else self.set_data(data)
        self.poly_array = np.zeros((self.n_faces, 2*self.n_face_nodes),
                                   dtype=np.float64)

    def set_data(self, data):
        '''Set Data Variable'''

        # Calculate face value based on # of face nodes
        self.face_array = np.mean(data[self.index], axis=1)

        # Replace previous face values
        if self.df is not None:
            self.df['faces'] = self.face_array.tolist()

    def to_poly_mesh(self):
        '''Construct Polygon Mesh'''
        p_index = np.arange(0, 2*self.n_face_nodes)

        # Vectorized Approach
        for n, i, j in zip(range(self.n_face_nodes), p_index[::2], p_index[1::2]):
            self.poly_array[:, i] = self.x[self.index[:, n]]
            self.poly_array[:, j] = self.y[self.index[:, n]]
    
        '''
        # Dask Parallel Approach
        poly_list_delayed = []    
        for n in range(self.n_faces):
            poly = delayed(self.create_poly)(n)
            poly_list_delayed.append(poly)

        poly_list = dask.compute(*poly_list_delayed)
        self.poly_array = np.array(poly_list)
        #'''



        # Create Polygon and Face Dataframe
        self.poly_array = self.poly_array.reshape((self.n_faces, 1, 8))
        polygons = sp.geometry.PolygonArray(self.poly_array.tolist())

        if self.face_array is None:
            self.df = sp.GeoDataFrame({'polygons': polygons})
        else:
            self.df = sp.GeoDataFrame({'polygons': polygons,
                                       'faces': self.face_array.tolist()})

    def create_poly(self, n):
         # [1 x n_face_nodes]
        poly_x = self.x[self.index[n]]
        poly_y = self.y[self.index[n]]

        # [1 x 2*n_face_nodes]
        poly_entry = np.insert(poly_x, np.arange(len(poly(x)), poly_y))
        return poly_entry


    def fix_cells(self):
        
        n_poly = self.df['polygons'].values.shape[0]
        poly_data = self.df['polygons'].values.buffer_values.reshape(n_poly, 8)
        face_data = self.df['faces'].values

        # Get x values
        x = poly_data[:, ::2]
        x_abs = np.abs(poly_data[:, ::2])

        # Find violater cells
        out_left = np.all((-x - x_abs) == 0, axis=1)
        out_right = np.all((x - x_abs) == 0, axis=1)
        out = out_left | out_right

        # Index of violater cells
        drop_index = np.arange(0, n_poly, 1)
        drop_index = drop_index[~out]

        # Get cells to fix
        poly_to_fix = poly_data[drop_index]

        # Exclude any center cells
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

        
        poly_insert = np.concatenate([poly_left_list, poly_right_list])
        face_insert = np.concatenate([face_list, face_list])

        poly_insert = poly_insert.reshape(2*len(poly_left_list), 1, 8)

        polygons = sp.geometry.PolygonArray(poly_insert.tolist())

        df_insert = sp.GeoDataFrame({'polygons': polygons,
                                    'faces': face_insert.tolist()})
        

        # Join edge and adjusted dataframes
        df_new = pd.concat([df_insert, df_droped], ignore_index=True)

        self.df = df_new
        return df_new

    def mesh(self):
        '''Return Polygon Mesh For Plotting with Datashader'''
        if self.df is None:
            return
        #df = self.fix_cells()
        return self.df
    
    def plot_data(self):
        '''not implemented'''
        return 
