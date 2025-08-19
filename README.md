# Get WorldPop Global Demographic Data

This repository contains some Python functions that can be used to acquire and summarise population cout from the WorldPop Global Demographic Dataset.

## Preparation
VRT file is used as the reference to multiple rasters in the dataset. `prep_script.py` can be used to create relevant VRT files and put them in `vrt` folder. Modify `dataset` value when needed.

In python console, we can run:
```python
exec(open('prep_script.py').read())
```

## Usage
```python
import get_wp_global as wp

# Obtaining population count inside non-overlapping
# circular buffers around points defined in adm.pkg
vrt_path = 'vrt/R2024B/mosaic_2020_100m_constrained.vrt'
result = wp.extract('adm.gpkg', vrt_path=vrt_path,
    rad=10, clip_buffer=True,
    return_gdf=True)

# Alternative usage
result = wp.get_data('adm.gpkg', dataset='R2024B', 
    year=2020, resolution='1km_ua', home_dir='vrt',
    return_gdf=False, rad=5, clip_buffer=False)
```