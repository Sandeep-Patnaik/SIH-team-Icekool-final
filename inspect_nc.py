import xarray as xr

# Open the NetCDF file
ds = xr.open_dataset("1900122_prof.nc")

# Basic information
print(ds)

print("\nVariables:")
print(list(ds.variables))

print("\nDimensions:")
print(ds.dims)

print("\nCoordinates:")
print(ds.coords)