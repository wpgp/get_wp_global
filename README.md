# Get WorldPop Global Demographic Data

This repository contains some Python functions that can be used to acquire and summarise population cout from the WorldPop Global Demographic Dataset.

## Usage
```python
import get_wp_global as wp

# Obtaining population count inside non-overlapping
# circular buffers around points defined in adm.pkg
result = wp.get_data('adm.gpkg', vrt_path='mosaic_2020.vrt',
    rad=10, clip_buffer=True,
    return_gdf=True)
```