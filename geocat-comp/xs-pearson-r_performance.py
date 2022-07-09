import pandas as pd
import xskillscore as xs
import xarray as xr
import numpy as np
n_lat = 90
n_lon = 180
n_time = 2

lat = np.linspace(-90, 90, n_lat)
lon = np.linspace(-180, 180, n_lon)
time = pd.date_range('2022-07-01', periods=n_time)
temp = 28 + 3 * np.random.randn(n_lon, n_lat, n_time)

ds = xr.Dataset(data_vars=dict(
                    temp=(['lon', 'lat', 'time'], temp)),
                coords=dict(
                    lon=lon,
                    lat=lat,
                    time=time,
                ),
                attrs=dict(description='Toy Weather Data'))

weights = np.cos(np.deg2rad(ds['lat']))

print(xs.pearson_r(ds.isel(time=0), ds.isel(time=1), dim='lat', weights=weights))