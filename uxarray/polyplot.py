import numpy as np
import xarray as xr
import pandas as pd
import dask as da
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
        for n, i, j in zip(range(self.n_face_nodes), p_index[::2], p_index[1::2]):
            self.poly_array[:, i] = self.x[self.index[:, n]]
            self.poly_array[:, j] = self.y[self.index[:, n]]

        # Create Polygon and Face Dataframe
        self.poly_array = self.poly_array.reshape((self.n_faces, 1, 8))
        polygons = sp.geometry.PolygonArray(self.poly_array.tolist())

        if self.face_array is None:
            self.df = sp.GeoDataFrame({'polygons': polygons})
        else:
            self.df = sp.GeoDataFrame({'polygons': polygons,
                                       'faces': self.face_array.tolist()})

    def fix_cells(self):
        '''Removes Edge Polygons'''
        n_poly = self.df['polygons'].values.shape[0]
        poly_data = self.df['polygons'].values.buffer_values.reshape(n_poly, 8)

        # Create mask for x coordinates of each polygon
        n = 2*self.n_face_nodes
        x_mask = [True if i % 2 == 0 else False for i in range(n)]

        # Find polygons with x values that are not the same sign
        out_positive = np.all(poly_data[:, x_mask] > -20, axis=1)
        out_negative = np.all(poly_data[:, x_mask] < 20, axis=1)
        out = np.logical_or(out_positive, out_negative)

        # Index for polygons to drop
        index = np.arange(0, n_poly, 1)
        index = index[np.invert(out)]

        # Create a new df with dropped polygons
        df = self.df.drop(index)
        return df

    def mesh(self):
        '''Return Polygon Mesh For Plotting with Datashader'''
        if self.df is None:
            return;
        df = self.fix_cells()
        return df
    
    def plot_data(self):
        hv.extension("bokeh")
        hv.output(dpi=300)
        cmap = cmap=cmocean.cm.curl

        projection = ccrs.PlateCarree(central_longitude=0)

        hvpolys = hv.Polygons(self.df, vdims=['faces']).opts(color='faces', tools=['hover'])
        rasterize(hvpolys, aggregator=ds.mean('faces')).opts(width=2000, height=1000, tools=['hover'], cmap=cmap, colorbar=True) * gf.coastline(projection=projection)
