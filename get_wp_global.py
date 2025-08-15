import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import pandas as pd

from tqdm import tqdm
from rasterio import features
from osgeo import gdal
from shapely.geometry import box
from typing import Dict, List, Tuple, Optional, Any, Callable

def _build_vrt(home_dir, prefix, suffix, outfile):
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
        file = f'{tlc.lower()}{suffix}.tif'
        dst = f'{home_dir}/{tlc}/{prefix}/{file}'
        path.append(dst)
        
    if not(outfile.endswith('vrt')):
        outfile += '.vrt'
        
    vrt = gdal.BuildVRT(outfile, path)
    vrt = None
    print('A new vrt file is created:', outfile)

def world_to_pixel(geotransform, x, y):
    """
    Converts world coordinates (x, y) to pixel/line indices.

    Args:
        geotransform (tuple): The 6-element geotransform of the dataset.
        x (float): The world x-coordinate.
        y (float): The world y-coordinate.

    Returns:
        tuple: A tuple containing the pixel and line index (px, py).
    """
    # Determinant
    det = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
    
    # Check for a valid geotransform
    if det == 0:
        raise ValueError("The geotransform determinant is zero, cannot invert.")

    px = (geotransform[5] * (x - geotransform[0]) - geotransform[2] * (y - geotransform[3])) / det
    py = (geotransform[1] * (y - geotransform[3]) - geotransform[4] * (x - geotransform[0])) / det

    return int(px), int(py)

def rasterise(v: gpd.GeoDataFrame, 
              t: np.ndarray, 
              nodata: Optional[float] = -1) -> np.ndarray:
    """
    Converts vector data to raster following a template raster.
    
    Args:
        - v: Vector data
        - t: Template raster in np.ndarray
        - nodata: No data value
        
    Returns:
        - Rasterised data in np.ndarray
    """
    
    bds = v.total_bounds
    shapes = ((geom, value) for geom, value in zip(v.geometry, v.index.values))
    transform = rasterio.transform.from_bounds(*bds, *t.shape[::-1])

    rst = features.rasterize(
        shapes,
        out_shape=t.shape,
        fill=nodata,
        all_touched=True,
        transform=transform,
        dtype=t.dtype
    )

    return rst

def get_raster_stats(t: np.ndarray,
                     m: np.ndarray,
                     ids: Optional[list] = None,
                     nodata: Optional[float] = None,
                     skip: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate statistics for a raster within mask regions.

    Args:
        t: Target raster data
        m: Mask raster data
        nodata: No data value
        skip: Value to skip in mask

    Returns:
        DataFrame with statistics
    """
    if ids is None:
        ids = np.unique(m)
    df_list = []

    for i in ids:
        if i == skip:
            continue

        select = np.logical_and(t != nodata, m == i)
        count = np.sum(select)
        if count < 1:
            continue

        tm = np.where(select, t, np.nan)
        d = pd.DataFrame({
            'id': i,
            'sum': np.nansum(tm)
        }, index=[0])

        df_list.append(d)

    if df_list:
        return pd.concat(df_list, ignore_index=True)

    return pd.DataFrame()

def get_buffer(v: gpd.GeoDataFrame,
               **kwargs) -> gpd.GeoDataFrame:
    '''
    Creates circular buffers around points defined
    in the input GeoDataFrame.
    
    Args:
        - v: GeoDataFrame defining the points of interest.
        - **kwargs: Additional arguments like 'rad' (defining
                    the buffer radius in km) and 'clip_buffer'
                    (whether to clip overlapping buffers or not).
    
    Returns:
        GeoDataFrame containing circular buffers.
    '''
    
    from scipy.spatial import Voronoi
    from shapely.geometry import LineString
    from shapely.ops import polygonize, linemerge, unary_union

    def get_voronoi(gdf_):
        # Creates voronoi tasselations based on the
        # points provided in the input GeoDataFrame. 

        vor = None
        bds = gdf_.total_bounds
        cx = gdf_['lon']
        cy = gdf_['lat']

        x1,y1 = bds[0]-10, bds[1]-10
        x2,y2 = bds[2]+10, bds[3]+10
        bounds = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
        coords = np.concatenate((
            np.stack([cx.tolist(), cy.tolist()], axis=1), bounds)
        )
        v = Voronoi(coords)
        lines = [LineString(v.vertices[line]) for line in 
                 v.ridge_vertices if -1 not in line]
        vor = gpd.GeoDataFrame(geometry=lines)
        return vor

    def non_overlaps(geom, line):
        # Clips geometry (geom) with lines (line)
        # where the original centroid is inside the
        # clipped geometry.

        line.append(geom.boundary)
        centroid = geom.centroid
        merged = linemerge(line)
        borders = unary_union(merged)
        polygons = np.array(list(polygonize(borders)))
        is_inside = [centroid.within(g) for g in polygons]

        return polygons[is_inside][0]

        return v_

    if 'rad' in kwargs:
        rad = kwargs.get('rad')
    else:
        rad = 5
        
    circ = v.to_crs(3857).buffer(1000*rad).to_crs(4326)
    if kwargs.get('clip_buffer'):
        voro = get_voronoi(v)
        bounds = circ.bounds
        for i,geom in enumerate(circ):
            b0 = bounds.iloc[i].tolist()
            g1 = voro.cx[b0[0]:b0[2], b0[1]:b0[3]]
            if (len(g1) > 0):
                non = non_overlaps(geom, g1.geometry.tolist())
                v.loc[i,'geometry'] = non
            else:
                v.loc[i,'geometry'] = geom
        v = v.reset_index(drop=True)
    else:
        v.geometry = circ.geometry
            
    return v

def get_data(gdf: gpd.GeoDataFrame, 
             vrt_path: str, 
             return_gdf: Optional[bool] = False, 
             **kwargs):
    '''
    Extracts total population in regions defined in the gdf.
    
    Args:
        - gdf: GeoDataFrame defining the regions of interest.
               It may contain POINTs, POLYGONs or MULTYPOLYGONs.
               If GeoDataFrame with POINTs is provided, then
               circular buffers around the POINTs are created
               and zonal statistics is performed based on these
               circluar buffers. The buffer radius and whether
               the buffers are clipped can be defined in the
               kwargs. By default, unclipped 5-km circular
               buffers are created.
               This argument can also be a path to GPKG, GeoJSON,
               SHP, or CSV file. If a CSV file is supplied, it
               should contain 'lon' and 'lat' column defining
               the locations of interest.
        - vrt_path: Path to GDAL virtual file defining the
                    raster mosaic.
        - return_gdf: Boolean to return
        - **kwargs: Additional arguments like 'rad' (defining
                    the buffer radius in km) and 'clip_buffer'
                    (whether to clip overlapping buffers or not).
                    
    Returns:
        DataFrame or GeoDataFrame with 'pop' (total population)
        column.
    
    Example:
        result = get_data('adm.gpkg', vrt_path='mosaic.vrt',
                          rad=10, clip_buffer=True,
                          return_gdf=True)
        print(type(result))
        #gpd.GeoDataFrame
    '''
    
    if gdf is str:
        ext = gdf.split('.')[-1]
        if ext in ['shp', 'gpkg', 'geojson']:
            gdf = gpd.read_file(gdf)
        elif ext in ['csv']:
            df = pd.read_csv(gdf)
            try:
                geo = gpd.points_from_xy(df.lon, df.lat, crs=4326)
                gdf = gpd.GeoDataFrame(geometry=geo)
                gdf['lon'] = df.lon
                gdf['lat'] = df.lat
            except:
                print('input csv does not contain lon-lat')
        else:
            print('can not read the input')
            return
        
    if (np.all(gdf.geometry.geom_type == 'Point')):
        gdf = get_buffer(gdf, **kwargs)
    bds = gdf.total_bounds
        
    bds_ = gpd.GeoDataFrame(geometry=[box(*bds)], crs=4326)
    area = 1e-6*bds_.to_crs(3857).area[0]
    if area > 1e7:
        print('parallelize please')
    
    vrt = gdal.Open(vrt_path)
    pop = vrt.GetRasterBand(1)
    trf = vrt.GetGeoTransform()
    ll = world_to_pixel(trf, bds[0], bds[1])
    ur = world_to_pixel(trf, bds[2], bds[3])
    dx,dy = ur[0]-ll[0], ll[1]-ur[1]
    pop_data = pop.ReadAsArray(xoff=ll[0], yoff=ur[1], win_xsize=dx, win_ysize=dy)
    nodata = pop.GetNoDataValue()
    rast = rasterise(gdf, pop_data, nodata=nodata)
    stat = get_raster_stats(pop_data, rast, 
                            nodata=nodata,
                            ids=gdf.index.values)
    stat = stat.rename(columns={'sum':'pop'})
    if return_gdf:
        gdf['pop'] = stat['pop']
        return gdf
    else:
        return stat
    
if __name__ == '__main__':
    sys.exit(get_data())