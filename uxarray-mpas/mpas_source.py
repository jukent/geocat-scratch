# source https://github.com/MPAS-Dev/MPAS-Tools
import sys
import netCDF4
import numpy as np
from optparse import OptionParser

from constants import *


def scrip_from_mpas_xr(mpas_ds, useLandIceMask=False):
    # Get info from input dataset
    latCell = mpas_ds['latCell'].values
    lonCell = mpas_ds['lonCell'].values
    latVertex = mpas_ds['latVertex'].values
    lonVertex = mpas_ds['lonVertex'].values
    verticesOnCell = mpas_ds['verticesOnCell'].values - 1
    nEdgesOnCell = mpas_ds['nEdgesOnCell'].values
    nCells = len(mpas_ds['nCells'].values)
    maxVertices = len(mpas_ds['maxEdges'].values)
    areaCell = mpas_ds['areaCell'].values
    
    #sphereRadius = float(fin.sphere_radius)
    #on_a_sphere = str(fin.on_a_sphere)

    #areaCell = fin.variables['areaCell'][:]
    


def scrip_from_mpas(mpasFile, scripFile, useLandIceMask=False):
    """
    Create a SCRIP file from an MPAS mesh

    Parameters
    ----------
    mpasFile : str
        The path to a NetCDF file with the MPAS mesh

    scripFile : str
        The path to the output SCRIP file

    useLandIceMask : bool
        Whether to use the landIceMask field for masking
    """
    if useLandIceMask:
        print(" -- Landice Masks are enabled")
    else:
        print(" -- Landice Masks are disabled")

    # make a space in stdout before further output
    print('')
    fin = netCDF4.Dataset(mpasFile, 'r')
    # This will clobber existing files
    fout = netCDF4.Dataset(scripFile, 'w')

    # Get info from input file
    latCell = fin.variables['latCell'][:]
    lonCell = fin.variables['lonCell'][:]
    latVertex = fin.variables['latVertex'][:]
    lonVertex = fin.variables['lonVertex'][:]
    verticesOnCell = fin.variables['verticesOnCell'][:] - 1
    nEdgesOnCell = fin.variables['nEdgesOnCell'][:]
    nCells = len(fin.dimensions['nCells'])
    maxVertices = len(fin.dimensions['maxEdges'])
    sphereRadius = float(fin.sphere_radius)
    on_a_sphere = str(fin.on_a_sphere)

    areaCell = fin.variables['areaCell'][:]

    # check the longitude convention to use positive values [0 2pi]
    if np.any(np.logical_or(lonCell < 0, lonCell > 2.0 * np.pi)):
       raise ValueError("lonCell is not in the desired range (0, 2pi)")

    if np.any(np.logical_or(lonVertex < 0, lonVertex > 2.0 * np.pi)):
       raise ValueError("lonVertex is not in the desired range (0, 2pi)")

    if sphereRadius <= 0:
       sphereRadius =  constants['SHR_CONST_REARTH']
       print(f" -- WARNING: sphereRadius<0 so setting sphereRadius = "
             f"{constants['SHR_CONST_REARTH']}")

    if on_a_sphere == "NO":
        print(" -- WARNING: 'on_a_sphere' attribute is 'NO', which means that "
              "there may be some disagreement regarding area between the "
              "planar (source) and spherical (target) mesh")

    if useLandIceMask:
        landIceMask = fin.variables['landIceMask'][:]
    else:
        landIceMask = None

    # Write to output file
    # Dimensions
    fout.createDimension("grid_size", nCells)
    fout.createDimension("grid_corners", maxVertices)
    fout.createDimension("grid_rank", 1)

    # Variables
    grid_center_lat = fout.createVariable('grid_center_lat', 'f8',
                                          ('grid_size',))
    grid_center_lat.units = 'radians'
    grid_center_lon = fout.createVariable('grid_center_lon', 'f8',
                                          ('grid_size',))
    grid_center_lon.units = 'radians'
    grid_corner_lat = fout.createVariable('grid_corner_lat', 'f8',
                                          ('grid_size', 'grid_corners'))
    grid_corner_lat.units = 'radians'
    grid_corner_lon = fout.createVariable('grid_corner_lon', 'f8',
                                          ('grid_size', 'grid_corners'))
    grid_corner_lon.units = 'radians'
    grid_imask = fout.createVariable('grid_imask', 'i4', ('grid_size',))
    grid_imask.units = 'unitless'
    grid_dims = fout.createVariable('grid_dims', 'i4', ('grid_rank',))

    grid_area = fout.createVariable('grid_area', 'f8', ('grid_size',))
    grid_area.units = 'radian^2'

    grid_center_lat[:] = latCell[:]
    grid_center_lon[:] = lonCell[:]
    grid_dims[:] = nCells

    #SCRIP uses square radians
    grid_area[:] = areaCell[:]/(sphereRadius**2)

    # grid corners:
    grid_corner_lon_local = np.zeros((nCells, maxVertices))
    grid_corner_lat_local = np.zeros((nCells, maxVertices))
    cellIndices = np.arange(nCells)
    lastValidVertex = verticesOnCell[cellIndices, nEdgesOnCell-1]
    for iVertex in range(maxVertices):
        mask = iVertex < nEdgesOnCell
        grid_corner_lat_local[mask, iVertex] = \
            latVertex[verticesOnCell[mask, iVertex]]
        grid_corner_lon_local[mask, iVertex] = \
            lonVertex[verticesOnCell[mask, iVertex]]

        mask = iVertex >= nEdgesOnCell
        grid_corner_lat_local[mask, iVertex] = latVertex[lastValidVertex[mask]]
        grid_corner_lon_local[mask, iVertex] = lonVertex[lastValidVertex[mask]]

    if useLandIceMask:
        # If useLandIceMask are enabled, mask out ocean under land ice.
        grid_imask[:] = 1 - landIceMask[0, :]
    else:
        # If landiceMasks are not enabled, don't mask anything out.
        grid_imask[:] = 1

    grid_corner_lat[:] = grid_corner_lat_local[:]
    grid_corner_lon[:] = grid_corner_lon_local[:]

    print("Input latCell min/max values (radians): {}, {}".format(
        latCell[:].min(), latCell[:].max()))
    print("Input lonCell min/max values (radians): {}, {}".format(
        lonCell[:].min(), lonCell[:].max()))
    print("Calculated grid_center_lat min/max values (radians): {}, {}".format(
         grid_center_lat[:].min(), grid_center_lat[:].max()))
    print("Calculated grid_center_lon min/max values (radians): {}, {}".format(
        grid_center_lon[:].min(), grid_center_lon[:].max()))

    print("Calculated grid_area min/max values (sq radians): {}, {}".format(
        grid_area[:].min(), grid_area[:].max()))

    fin.close()
    fout.close()

    print("Creation of SCRIP file is complete.")