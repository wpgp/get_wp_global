import os
import sys
import requests
import argparse
import numpy as np

notes = {
    'pop': 'total population count',
    'male': 'male population per age group',
    'female': 'female population per age group',
    'zip': 'zipped age-sex structure data'
}
available_layers = list(notes.keys())
all_tlcs = ['ABW', 'AFG', 'AGO', 'AIA', 'ALA', 'ALB', 'AND',
       'ARE', 'ARG', 'ARM', 'ASM', 'ATF', 'ATG', 'AUS', 'AUT', 'AZE',
       'BDI', 'BEL', 'BEN', 'BES', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS',
       'BIH', 'BLM', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN',
       'BTN', 'BVT', 'BWA', 'CAF', 'CAN', 'CCK', 'CHE', 'CHL', 'CHN',
       'CIV', 'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPT', 'CPV',
       'CRI', 'CUB', 'CUW', 'CXR', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI',
       'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESH', 'ESP',
       'EST', 'ETH', 'FIN', 'FJI', 'FLK', 'FRA', 'FRO', 'FSM', 'GAB',
       'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN', 'GLP', 'GMB', 'GNB',
       'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUF', 'GUM', 'GUY', 'HKG',
       'HMD', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IND', 'IOT',
       'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JEY', 'JOR',
       'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT',
       'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO', 'LTU',
       'LUX', 'LVA', 'MAC', 'MAF', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV',
       'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP',
       'MOZ', 'MRT', 'MSR', 'MTQ', 'MUS', 'MWI', 'MYS', 'MYT', 'NAM',
       'NCL', 'NER', 'NFK', 'NGA', 'NIC', 'NIU', 'NLD', 'NOR', 'NPL',
       'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PCN', 'PER', 'PHL', 'PLW',
       'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT',
       'REU', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SGS',
       'SHN', 'SJM', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SPM', 'SRB',
       'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM', 'SYC',
       'SYR', 'TCA', 'TCD', 'TGO', 'THA', 'TJK', 'TKL', 'TKM', 'TLS',
       'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR',
       'UMI', 'URY', 'USA', 'USA_States', 'UZB', 'VAT', 'VCT', 'VEN',
       'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'XDI', 'XIB', 'XIK',
       'XKX', 'XMA', 'XSI', 'YEM', 'ZAF', 'ZMB', 'ZWE']
all_tlcs = ['ABW', 'AFG']

def show_available_layers():
    print('The following layers are available for download:')    
    for n in notes:
        print(f'- {n}: {notes[n]}')
    return
    
def check_nearest_tlc():
    print('Check tlc spelling.')
    a1 = [a[:2]==tlc[:2] for a in all_tlcs]
    a2 = [a[-2:]==tlc[-2:] for a in all_tlcs]

    b1 = np.array(all_tlcs)[a1]
    b2 = np.array(all_tlcs)[a2]
    
    print('Closest resemblance:')
    print(np.append(b1, b2))
    return
    
def get_urls(layer='pop', tlc='MOS', year=2020, res='1km',
            dataset='R2025A', age_range=[0,10], **kwargs):
    """
    Locates path to raster related to Worldpop Global Demographic Data
    (Global 2)
    
    Args:
        layer (str): Selected layer to download. Use 'available_layer()'
            to see available options.
        tlc (str): Three letter code (alpha-3, ISO 3166-1) of the country
            to download. Use 'ALL' to get paths of all available countries.
            Use 'MOS' to get global mosaic (only for 1km resolution).
        year (int): Year of the data. Use '0' to download all data (from 2015
            to 2030).
        res (str): Spatial resolution of the raster ('100m' or '1km').
        dataset (str): Dataset code
        age_range (list): Range of the age to get
    """
    
    if not(layer in available_layers):
        print('Layer you select is not available.')
        show_available_layers()
        return
    
    if tlc not in all_tlcs+['MOS', 'ALL']:
        print('Three letter code (tlc) not found.')
        print("Use 'ALL' to get paths of all available countries.")
        print("Use 'MOS' to get global mosaic (only for 1km resolution).")
        check_nearest_tlc(tlc)
        return
    
    if (year > 2030) or (year < 2015):
        print('Selected year should be between 2015 and 2030 (inclusive).')
        return
    
    if res not in ['1km', '100m']:
        print('Available resolution: 100m, 1km')
        return
    
    base = 'https://data.worldpop.org/GIS'
    age_groups = []
    agesex = None
    paths = []
    suffix = 'v1.tif'
    res_ = res
    
    if layer in ['male', 'female', 'zip']:
        base += f'/AgeSex_structures/Global_2015_2030/{dataset}'
        age_groups = list(range(age_range[0], age_range[1]+1, 5))
        if (age_range[0] < 1) & (age_range[1] > 1):
            age_groups.insert(1, 1)
        
        if layer == 'male':
            agesex = [f'm_{a:02d}' for a in age_groups]
        if layer == 'female':
            agesex = [f'f_{a:02d}' for a in age_groups]
        if layer == 'zip':
            agesex = ['agesex_structures']
            age_groups = list(range(0, 91, 5))
            age_groups.insert(1,1)

        print('Available age groups:')
        print(age_groups)
            
    if layer in ['pop']:
        base += f'/Population/Global_2015_2030/{dataset}'
    
    if year == 0:
        year = [str(a) for a in range(2015,2031)]
    else:
        year = [str(year)]
    
    if tlc == 'ALL':
        tlc = all_tlcs
        name = [a.lower() for a in tlc]
    elif tlc == 'MOS':
        tlc = ['0_Mosaicked']
        name = ['global']
        res = '1km'
        res_ = '1km'
    else:
        tlc = [tlc]
        name = [a.lower() for a in tlc]        
    
    if res == '1km':
        res = '1km_ua'
        suffix = 'UA_v1.tif'
    else:
        res = '100m'
        
    for y in year:
        p = f'{base}/{y}'
        for t,n in zip(tlc,name):
            fname = f'{n}_{layer}_{y}_CN_{res_}_{dataset}_{suffix}'
            paths.append(f'{p}/{t}/v1/{res}/constrained/{fname}')
            
    if agesex is not None:
        paths_ = paths.copy()
        paths = []
        for p in paths_:
            for a in agesex:
                paths.append(p.replace(layer, a))
                
    if layer == 'zip':
        paths = [p.replace('tif','zip').replace('constrained/','') for p in paths]
                
    return paths

def download_file(url, destination_folder, filename=None):
    """
    Downloads a file from a given URL to a specified destination folder.

    Args:
        url (str): The URL of the file to download.
        destination_folder (str): The path to the folder where the file will be saved.
        filename (str, optional): The name to save the file as. If None,
                                  the original filename from the URL will be used.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory: {destination_folder}")

    if filename is None:
        filename = url.split('/')[-1] # Extract filename from URL

    filepath = os.path.join(destination_folder, filename)

    try:
        print(f"Downloading {url} to {filepath}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete!")
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        
def usage():
    print('Usage:')
    print()
    
def wrapper():
    parser = argparse.ArgumentParser(
        prog="download_wp_global",
        description="Simple program to download Worldpop Global"
            "Demographic Data to local storage."
    )
    
    parser.add_argument('-l', '--layer', help="selected layer to download [pop, female, male, zip]",
                        type=str, default='pop')
    parser.add_argument('-t', '--tlc', help="three letter code of the country to download", 
                        type=str, default='MOS')
    parser.add_argument('-d', '--dataset', help="dataset number",
                        type=str, default='R2025A')
    parser.add_argument('-y', '--year', help="year", 
                        type=int, default=2020)
    parser.add_argument('-ar', '--age_range', help="min and max age group to download, separated by comma",
                        type=str, default='0,10')
    parser.add_argument('-dst', '--destination', help="destination folder", 
                        type=str, default='./')
    
    args = vars(parser.parse_args())
    if not(os.path.exists(args['destination'])):
        os.makedirs(args['destination'])
        
    if 'age_range' in args.keys():
        a = args['age_range'].replace(' ','').split(',')
        args['age_range'] = [int(b) for b in a]

    urls = get_urls(**args)
    for url in urls:
        download_file(url, args['destination'])
    
if __name__ == '__main__':
    sys.exit(wrapper())