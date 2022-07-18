import pandas as pd
import xskillscore as xs
import xarray as xr
import numpy as np
import time

def create_data(n_lat, n_lon, n_time):
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
    return ds, weights


def test(n_lat, n_lon, n_time):
    ds, w = create_data(n_lat, n_lon, n_time)
    start = time.perf_counter()
    xs.pearson_r(ds.isel(time=0), ds.isel(time=1), dim='lat', weights=w)
    return time.perf_counter()-start

times= [test(1000,1000,2),
        test(10000,1000,2),
        test(10000,10000,2),
        test(100000,10000,2)]

num_pts = [1000*1000*2, 10000*1000*2, 10000*10000*2, 100000*10000*2]
import matplotlib.pyplot as plt
plt.plot(num_pts, times)
plt.show()

from scipy.stats import pearsonr

print(pearsonr(num_pts, times))