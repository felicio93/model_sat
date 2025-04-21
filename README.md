# model_sat

Download, crop, and collocate satellite wave height data with model outputs (e.g. SCHISM).

## Install

```
pip install git+https://github.com/felicio93/model_sat
```
## Usage
```
import numpy as np
import xarray as xr
import ocsmesh 
from model_sat import collocate, get_sat

ds_sat = get_sat.get_multi_sat(
                               start_date="2019-06-01",
                               end_date="2019-06-30",
                               sat=['sentinel3a','sentinel3b','jason2','jason3','cryosat2','saral'],
                               output_dir="./sat_data",
                               lat_min=49.109,
                               lat_max=66.304309,
                               lon_min=156.6854,
                               lon_max=-156.864,
                               )


rundir = r'/Your/SCHISM/RunDir/R0*/'
model_paths = collocate.select_model_files_in_timerange(rundir, np.datetime64(start_date), np.datetime64(end_date))
mesh = ocsmesh.Mesh.open(rundir + 'hgrid.gr3', crs=4326)
dist_coast = xr.open_dataset(r'/Your/Path/to/distFromCoast.nc')

mesh_x = collocate.convert_longitude(mesh.vert2['coord'][:, 0], 2) #this is necessary if the mesh's lon is not within -180 to 180
mesh_y = mesh.vert2['coord'][:, 1]
mesh_depth = mesh.value.ravel()
collocated = collocate.collocate_data(model_paths,
                                      variable_name,
                                      ds_sat,
                                      mesh_x,
                                      mesh_y,
                                      mesh_depth,
                                      dist_coast,
                                      model,
                                      n_nearest=3,
                                      time_buffer=np.timedelta64(30, 'm'),
                                      weight_power=1.0,
                                      temporal_interp=True,
                                      output_path=output_dir+"collocated.nc")
```
