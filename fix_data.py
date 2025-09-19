from netCDF4 import Dataset
import numpy as np

# Input and output file names
input_file = "./data/fictional_pm10_3d_50x50x15_timelapse.nc"
output_file = "./data/fictional_pm10_3d_50x50x15_timelapse_fixed.nc"

# Open original file in read mode
with Dataset(input_file, "r") as src:
    # Create new NetCDF file
    with Dataset(output_file, "w", format=src.file_format) as dst:
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        
        # Copy all variables except 'y'
        for name, var in src.variables.items():
            if name != 'y':
                out_var = dst.createVariable(name, var.datatype, var.dimensions)
                # Copy variable attributes
                out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                # Copy data
                out_var[:] = var[:]
        
        # Create new 'y' variable as double precision
        y_dim_size = len(src.dimensions['y'])
        y_vals = 6700000.0 + 0.5 * np.arange(y_dim_size, dtype=np.float64)
        y_var = dst.createVariable('y', 'f8', ('y',))
        # Add attributes (you can copy original ones if needed)
        y_var.long_name = "projection_y_coordinate"
        y_var.units = "m"
        y_var.axis = "Y"
        y_var[:] = y_vals
        
        # Copy global attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

print(f"Created fixed NetCDF file: {output_file}")
