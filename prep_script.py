import get_wp_global as wp
import os

dataset = 'R2024B'
home_dir = 'Z:/WPFTP/public/GIS/Population/Global_2015_2030'

if not(os.path.isdir('vrt')):
    os.mkdir('vrt')
if not(os.path.isdir(f'vrt/{dataset}'):
    os.mkdir(f'vrt/{dataset}')

for res in ['100m', '1km_ua']
    prefix = f'v1/{res}/constrained'
    for y in range(2017, 2031, 1):
        main_dir = f'{home_dir}/{dataset}/{y}'
        if res == '1km_ua':
            suffix = f'_pop_{y}_CN_100m_{dataset}_UA_v1'
        else:
            suffix = f'_pop_{y}_CN_100m_{dataset}_v1'
        outfile = f'vrt/{dataset}/mosaic_{y}_{res}_constrained.vrt'
        wp._build_vrt(home_dir, prefix, suffix, outfile)
        