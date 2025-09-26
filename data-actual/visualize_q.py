import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# Load NetCDF
nc_file = 'C:/Users/ramic/Nansen/data/OUTPUT/dp_3d.001.nc' # Update with your actual path
with Dataset(nc_file) as nc:
    q = nc.variables['q'][0] # q represents air moisture levels
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    z = nc.variables['zu_3d'][:] # Vertical levels at the center of the u, v, theta, and q variables (model’s main or “scalar” levels).

# Downsample
x_ds = x[::2]
y_ds = y[::2]
z_ds = z[::2]
q_ds = q[::2, ::2, ::2]  # shape (z, y, x)

# Correct meshgrid to match Q shape (z, y, x)
Zg, Yg, Xg = np.meshgrid(z_ds, y_ds, x_ds, indexing='ij')

# Flatten arrays
Xf = Xg.flatten()
Yf = Yg.flatten()
Zf = Zg.flatten()
Cf = q_ds.flatten()

# Normalize Q values for color and transparency
norm = cm.colors.Normalize(vmin=Cf.min(), vmax=Cf.max())
cmap = cm.viridis
colors = cmap(norm(Cf))  # RGBA colors
colors[:, -1] = norm(Cf)  # set alpha proportional to Q

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
fig.colorbar(mappable, ax=ax, label='Q (kg/kg(?))')

plt.show()
