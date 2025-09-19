import numpy as np
import pyvista as pv
from netCDF4 import Dataset

# Load the NetCDF data
nc_file = './data/fictional_pm10_3d_50x50x15_timelapse_fixed.nc'

with Dataset(nc_file) as nc:
    # Take the first time step
    pm10 = nc.variables['kc_PM10'][0, :, :, :]  # shape (z, y, x)
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    z = nc.variables['z'][:]  # height above ground

# Create a 3D structured grid
# PyVista expects coordinates in 3D arrays: shape (nx, ny, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (nx, ny, nz)

# Create the structured grid
grid = pv.StructuredGrid(X, Y, Z)

# Flatten PM10 data to match the structured grid ordering
grid["PM10"] = np.transpose(pm10, (2, 1, 0)).flatten(order="F")  # shape (nx*ny*nz)

# Create a PyVista plotter
plotter = pv.Plotter()
plotter.add_volume(
    grid,
    scalars="PM10",
    cmap="viridis",
    opacity="linear",  # linear opacity mapping
    shade=True,
)

# Add axes and show the interactive plot
plotter.show_axes()
plotter.show()
