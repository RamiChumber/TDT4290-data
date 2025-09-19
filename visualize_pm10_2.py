import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# Load NetCDF
nc_file = './data/fictional_pm10_3d_50x50x15_timelapse_fixed.nc'
with Dataset(nc_file) as nc:
    pm10 = nc.variables['kc_PM10'][0]  # first time step
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    z = nc.variables['z'][:]

# Downsample
x_ds = x[::2]
y_ds = y[::2]
z_ds = z[::2]
pm10_ds = pm10[::2, ::2, ::2]  # shape (z, y, x)

# Correct meshgrid to match PM10 shape (z, y, x)
Zg, Yg, Xg = np.meshgrid(z_ds, y_ds, x_ds, indexing='ij')

# Flatten arrays
Xf = Xg.flatten()
Yf = Yg.flatten()
Zf = Zg.flatten()
Cf = pm10_ds.flatten()

# Normalize PM10 values for color and transparency
norm = cm.colors.Normalize(vmin=Cf.min(), vmax=Cf.max())
cmap = cm.viridis
colors = cmap(norm(Cf))  # RGBA colors
colors[:, -1] = norm(Cf)  # set alpha proportional to PM10

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(Xf, Yf, Zf, c=colors, marker='o', s=5)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')

# Add a colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(Cf)
fig.colorbar(mappable, ax=ax, label='PM10 (kg/m³)')

plt.show()
