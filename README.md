# Get WorldPop Global Demographic Data

This repository contains some Python functions that can be used to acquire and summarise population cout from the WorldPop Global Demographic Dataset.

## Preparation
VRT file is used as the reference to multiple rasters in the dataset. `prep_script.py` can be used to create relevant VRT files and put them in `vrt` folder. Modify `dataset` value when needed.

In python console, we can run:
```python
exec(open('prep_script.py').read())
```

## Usage
### Population count
Obtaining population count inside non-overlapping circular buffers around points defined in `adm.pkg`.

```python
import get_wp_global as wp

vrt_path = 'vrt/R2024B/mosaic_2020_100m_constrained.vrt'
result = wp.extract('adm.gpkg', vrt_path=vrt_path,
    rad=10, clip_buffer=True,
    return_gdf=True)

# Alternative usage
result1 = wp.get_data('adm.gpkg', dataset='R2024B', 
    year=2020, resolution='1km', vrt_dir='vrt',
    return_gdf=True, rad=5, clip_buffer=False)
```

### Age-sex structure
Extracting female population count with specified age range can be done using `get_data_agesex()`. The output contains population count at 5-year age interval. Total population count can also be extracted. This total covers the whole population, both sexes and all age intervals.

```python
import get_wp_global as wp

result2 = wp.get_data_agesex('adm.geojson', dataset='R2024B', 
    year=2020, resolution='1km', 
    vrt_dir='vrt', sex='female', get_total=True,
    return_gdf=False)

result2.head()
```

|    |   id |  f_00 |  f_05 |  f_10 |     pop |
|---:|-----:|------:|------:|------:|--------:|
|  0 |    0 | 18161 | 76259 | 65505 | 1087010 |
|  1 |    1 |  2291 |  9623 |  8266 |  137168 |
|  2 |    2 |  1428 |  5996 |  5151 |   85479 |
|  3 |    3 |  4200 | 17637 | 15150 |  251403 |
|  4 |    4 |   296 |  1244 |  1068 |   17732 |

### Some visualisations

Extracting gridded population count based on level-2 administrative boundaries covering some parts of Ghana, Benin, and Togo. Zonal statistics can be performed to obtain total population inside each administrative unit.

![map](fig/arr.png)

Extraction of total population using admin boundary (a) and circular buffer (b). The circular buffer is generated from the centroid of each administrative unit, which then clipped to avoid overlap.

![map](fig/res.png)