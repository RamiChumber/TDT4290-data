import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from netCDF4 import Dataset

# Load NetCDF
nc_file = './data/custom/fictional_pm10_3d_50x50x15_timelapse_fixed.nc'
with Dataset(nc_file) as nc:
    pm10 = nc.variables['kc_PM10'][:]  # shape: (time, z, y, x)
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    z = nc.variables['z'][:]

# Downsample for faster plotting
x_ds = x[::2]
y_ds = y[::2]
z_ds = z[::2]
pm10_ds = pm10[:, ::2, ::2, ::2]

Xg, Yg, Zg = np.meshgrid(x_ds, y_ds, z_ds, indexing='ij')
Xf = Xg.ravel()
Yf = Yg.ravel()
Zf = Zg.ravel()


# Color normalization
vmin, vmax = pm10_ds.min(), pm10_ds.max()
norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.viridis

size_min, size_max = 1, 8

def get_marker_size(Cf):
    norm_val = (Cf - vmin) / (vmax - vmin)
    return size_min + (size_max - size_min) * norm_val

x_min, x_max = x_ds.min(), x_ds.max()
y_min, y_max = y_ds.min(), y_ds.max()
z_min, z_max = z_ds.min(), z_ds.max()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

Cf0 = pm10_ds[0].flatten()
colors0 = cmap(norm(Cf0))[:, :3]
sizes0  = get_marker_size(Cf0)

ax.set_xlim(x_ds.min(), x_ds.max())
ax.set_ylim(y_ds.min(), y_ds.max())
ax.set_zlim(z_ds.min(), z_ds.max())


ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')

mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(pm10_ds)
fig.colorbar(mappable, ax=ax, label='PM10 (kg/m³)')

def clear_scatters():
    for coll in ax.collections[:]:
        coll.remove()

def update(frame):

    clear_scatters()

    frame_data = pm10_ds[frame]  # shape (z, y, x)

    # Flatten coordinates to match the array order
    Zf, Yf, Xf = np.meshgrid(z_ds, y_ds, x_ds, indexing='ij')
    Xf = Xf.ravel()
    Yf = Yf.ravel()
    Zf = Zf.ravel()
    Cf = frame_data.ravel()  # matches Z/Y/X ordering


    # Define PM10 bins and corresponding alpha values
    bins = [
        (vmin, vmin + 0.25*(vmax-vmin), 0.2),
        (vmin + 0.25*(vmax-vmin), vmin + 0.5*(vmax-vmin), 0.4),
        (vmin + 0.5*(vmax-vmin), vmin + 0.75*(vmax-vmin), 0.6),
        (vmin + 0.75*(vmax-vmin), vmax, 0.8)
    ]

    # Create scatter for each bin
    for lo, hi, a in bins:
        mask = (Cf >= lo) & (Cf < hi)
        if np.any(mask):
            ax.scatter(
                Xf[mask],
                Yf[mask],
                Zf[mask],
                c=cmap(norm(Cf[mask]))[:, :3],  # RGB only
                s=get_marker_size(Cf[mask]),
                alpha=a
            )

    ax.set_title(f"Time step: {frame}")
    return ax.collections




# Create animation
anim = FuncAnimation(fig, update, frames=pm10_ds.shape[0], interval=200, blit=False)

# Save to MP4
writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
anim.save('pm10_3d_animation.mp4', writer=writer)

plt.close(fig)
