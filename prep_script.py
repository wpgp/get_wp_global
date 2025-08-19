import os
from osgeo import gdal

dataset = 'R2024B'

'''
Expected directory tree:
vrt/[dataset]
|-- pop
|   |-- 1km_ua
|   |   |-- 2015
|   |   |   |-- mosaic_2015_1km_ua_constrained.vrt
|   |   |   |-- mosaic_2020_100m_constrained.vrt
|   |   |-- 2016
|   |   ...
|   |   |-- 2030
|   |-- 100m
|-- agesex
    |-- 1km_ua
    |   |-- 2015
    |   |   |-- mosaic_f_00_2015_1km_ua_constrained.vrt
    |   |   |-- mosaic_f_01_2015_1km_ua_constrained.vrt
    |   |   ...
    |   |   |-- mosaic_m_90_2015_1km_ua_constrained.vrt
    |   |-- 2016
    |   ...
    |   |-- 2030
    |-- 100m        
'''

def build_vrt(home_dir: str, 
    prefix: str, 
    suffix: str, 
    outfile: str):
    '''
    List available rasters with specified pattern and 
    build GDAL virtual file for mosaicing purpose.
    
    Args:
        - home_dir: Directory containing the country
                    folders (e.g., AFG)
        - prefix: Relative path from country folder
                  and the raster
        - suffix: File name suffix
        - outfile: Output file name
        
    prefix = 'v1/1km_ua/constrained'
    suffix = '_pop_2020_CN_1km_R2024B_UA_v1'
    '''
    
    tlcs = os.listdir(home_dir)
    path = []
    for tlc in tlcs:
        if len(tlc) > 3: continue
        file = f'{tlc.lower()}{suffix}.tif'
        dst = f'{home_dir}/{tlc}/{prefix}/{file}'
        path.append(dst)
        
    if not(outfile.endswith('vrt')):
        outfile += '.vrt'
        
    vrt = gdal.BuildVRT(outfile, path)
    vrt = None
    print('A new vrt file is created:', outfile)

#Preparing folders and subfolders
for y in range(2015, 2031):
    os.makedirs(f'vrt/{dataset}/pop/1km_ua/{y}', exist_ok=True)
    os.makedirs(f'vrt/{dataset}/pop/100m/{y}', exist_ok=True)
    os.makedirs(f'vrt/{dataset}/agesex/1km_ua/{y}', exist_ok=True)
    os.makedirs(f'vrt/{dataset}/agesex/100m/{y}', exist_ok=True)

#Creating virtual files for total population
home_dir = 'Z:/WPFTP/public/GIS/Population/Global_2015_2030'
home_dir = '../data'
for res in ['1km']:
    add = 'v1'
    folder = res
    if res == '1km':
        add = 'UA_v1'
        folder = '1km_ua'
    prefix = f'v1/{folder}/constrained'
    for y in range(2020, 2021):
        main_dir = f'{home_dir}/{dataset}/{y}'
        suffix = f'_pop_{y}_CN_{res}_{dataset}_{add}'
        outfile = f'vrt/{dataset}/pop/{folder}/{y}/mosaic_{y}_{res}_constrained.vrt'
        build_vrt(main_dir, prefix, suffix, outfile)
        
'''
#Creating virtual files for total age-sex structures
home_dir = 'Z:/WPFTP/public/GIS/AgeSex_structures/Global_2015_2030'
home_dir = '../data'
age_groups = list(range(0,15,5))
age_groups.insert(1, 1)

for res in ['100m', '1km']:
    add = 'v1'
    folder = res
    if res == '1km':
        add = 'UA_v1'
        folder = '1km_ua'
    prefix = f'v1/{folder}/constrained'
    for y in range(2015, 2031):
        main_dir = f'{home_dir}/{dataset}/{y}'
        for a in age_groups:
            #female
            suffix = f'_f_{a:02d}_{y}_CN_{res}_{dataset}_{add}'
            outfile = f'vrt/{dataset}/agesex/{folder}/{y}/mosaic_f_{a:02d}_{y}_{res}_constrained.vrt'
            build_vrt(main_dir, prefix, suffix, outfile)

            #male
            suffix = f'_m_{a:02d}_{y}_CN_{res}_{dataset}_{add}'
            outfile = f'vrt/{dataset}/agesex/{folder}/{y}/mosaic_m_{a:02d}_{y}_{res}_constrained.vrt'
            build_vrt(main_dir, prefix, suffix, outfile)        
'''