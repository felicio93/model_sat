# model_sat

Download, crop, and collocate satellite wave height data with model outputs (e.g. SCHISM).

## Install

```
pip install git+https://github.com/felicio93/model_sat
```
## Usage
```
from model_sat import get_sat

get_sat(
    start_date="2019-06-01",
    end_date="2019-06-30",
    sat='sentinel3a',
    output_dir="./sat_data",
    lat_min=49.0,
    lat_max=66.0,
    lon_min=156.0,
    lon_max=-156.0
)
```
