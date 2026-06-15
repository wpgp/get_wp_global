import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import utils

from rasterio import features
from osgeo import gdal
from shapely.geometry import box
from typing import List, Optional, Union

def world_to_pixel(
    geotransform, x, y):
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

def rasterise(
    v: gpd.GeoDataFrame, 
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

def get_raster_stats(
    t: np.ndarray,
    m: np.ndarray,
    w: Optional[np.ndarray] = None,
    nodata: Optional[float] = -1,
    skip: Optional[list] = None,
    names: Optional[list] = None) -> pd.DataFrame:
    """
    Calculate statistics for a raster within mask regions.

    Args:
        t: Target raster data
        m: Mask raster data
        w: Weight layer
        nodata: No data value
        skip: List of values to skip in mask

    Returns:
        DataFrame with statistics
    """
    if len(t.shape) == 2:
        t = t.reshape((1, *t.shape))

    if names is None:
        names = [f'sum_{a:02d}' for a in range(t.shape[0])]

    if w is None:
        v = t.copy()
    else:
        v = t*w

    v[t == nodata] = np.nan
    data = {'id':m.flatten()}
    for j in range(t.shape[0]):
        data[names[j]] = v[j,:,:].flatten()

    df = pd.DataFrame(data)
    if skip is not None:
        a = df['id'].isin(skip)
        df = df[~a]

    cnt = df[['id', names[0]]].groupby('id').agg('count').reset_index()
    out_df = df.groupby('id').agg('sum').reset_index()
    out_df['count'] = cnt[names[0]].values

    return out_df

def get_buffer(
    v: gpd.GeoDataFrame,
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

def get_extent(ds):
    transform = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Calculate bounds
    minx = transform[0]
    maxy = transform[3]
    maxx = minx + width * transform[1]
    miny = maxy + height * transform[5]
    return (minx, miny, maxx, maxy)

def get_mask(
    v: gpd.GeoDataFrame, 
    resolution: Optional[str] = '1km',
    ext_vrt: Optional[tuple] = None,
    **kwargs):
    
    if resolution == '100m':
        d = 0.00083333333
    if resolution in ['1km', '1km_ua', '1000m']:
        d = 0.0083333333

    bds = v.total_bounds

    #convert radius from km to pixel
    if 'rad' in kwargs:
        lat = 0.5*(bds[1]+bds[3])
        rad = np.round(np.degrees(kwargs['rad']/6378)/(d*np.cos(lat)))
    else:
        #use default radius of 5 pixels
        rad = 5

    #todo: get fun from kwargs if any
    fun = utils.default_function

    #kernel size should be odd
    size = int(2*rad)
    if size % 2 == 0:
        size += 1

    bds = utils.validate_extent(bds, resolution=resolution, 
                                ext_vrt=ext_vrt, buffer=rad)
    bds = list(bds)
    sx = np.round((bds[2] - bds[0])/d).astype(int)
    sy = np.round((bds[3] - bds[1])/d).astype(int)
    
    shp = ((geom, value) for geom, value in zip(v.geometry.values, v['id'].values))
    trf = rasterio.transform.Affine(d, 0, bds[0], 0, -d, bds[3])
    arr = features.rasterize(
        shp,
        out_shape=(sy, sx),
        fill=0,
        transform=trf,
        all_touched=True
    )
    
    krn = utils.radial_kernel(size, lambda r: np.where(r <= rad, float(1), 0.0))
    msk = utils.max_conv_2d(arr, krn, mode='same', weighted=True)
    m = msk == 0
    out = utils.fill_nearest(arr, m)
    
    #creating weight layer
    wgt = None
    if kwargs.get('weighted'):
        if 'p' not in kwargs:
            kwargs['p'] = 1
        krn = utils.radial_kernel(size, lambda r: fun(r, kwargs['p'], rnorm=rad))
        krn /= krn.max()
        wgt = utils.max_conv_2d(arr > 0, krn, mode='same', weighted=True)

    obj = {'mask':out, 'weight':wgt, 'kernel':krn, 'bounds':bds, 'transform':trf}
    return obj

def validate_vector_data(
    vec, **kwargs):

    if isinstance(vec, str):
        ext = vec.split('.')[-1]
        if ext in ['shp', 'gpkg', 'geojson']:
            gdf = gpd.read_file(vec)
        elif ext in ['csv']:
            df = pd.read_csv(vec)
            try:
                geo = gpd.points_from_xy(df.lon, df.lat, crs=4326)
                gdf = gpd.GeoDataFrame(geometry=geo)
            except:
                print('input csv does not contain lon-lat')
        else:
            print('can not read the input')
            return
        
    if kwargs.get('explode'):
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

    if kwargs.get('edge') and np.all(gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])):
        geom = gdf.boundary.geometry.values
        gdf['geometry'] = geom

    gdf['id'] = gdf.index.values + 1
    return gdf

def validate_raster_data(
    src: list, **kwargs):
    
    ext = src[0].split('.')[-1]
    if not(os.path.isfile(src[0])):
        print('file not found\n', src[0])
        return

    if ext in ['tif']:
        with rasterio.open(src[0]) as s:
            nodata = s.nodata
            trf = s.transform
            bds = list(s.bounds)
            data_type = 'raster'
    elif ext in ['vrt']:
        vrt = gdal.Open(src[0])
        pop = vrt.GetRasterBand(1)
        trf = vrt.GetGeoTransform()
        nodata = pop.GetNoDataValue()
        bds = get_extent(vrt)
        data_type = 'vrt'
        vrt = None
    else:
        print('input file should either be: *.tif, *.vrt')
        return

    obj = {'extent':bds, 'transform':trf, 'nodata':nodata, 'data_type':data_type}
    return obj

def extract(
    vec: Union[gpd.GeoDataFrame, str], 
    rst: Union[str, list],
    names: Optional[list] = ['pop'],
    resolution: Optional[str] = '1km',
    **kwargs):

    #Check vector data input
    if isinstance(rst, str):
        rst = [rst]

    gdf = validate_vector_data(vec)
    par = validate_raster_data(rst)
    bds = gdf.total_bounds

    msk = get_mask(gdf, ext_vrt=par['extent'], **kwargs)
    bds = msk['bounds']

    ll = world_to_pixel(par['transform'], bds[0], bds[1])
    ur = world_to_pixel(par['transform'], bds[2], bds[3])
    sx = msk['mask'].shape[1]
    sy = msk['mask'].shape[0]

    #obtain data stack
    data = []
    if par['data_type'] == 'raster':
        for path in rst:
            with rasterio.open(path) as s:
                ds = s.read(1)
            data.append(ds)
    else:
        for path in rst:
            vrt = gdal.Open(path)
            ds = vrt.GetRasterBand(1)
            ar = ds.ReadAsArray(xoff=ll[0], yoff=ur[1], win_xsize=sx, win_ysize=sy)
            data.append(ar)
            vrt = None
    data = np.array(data)

    #perform zonal statistics
    stat = get_raster_stats(data, 
                            msk['mask'],
                            w=msk['weight'], 
                            nodata=par['nodata'],
                            skip=[0], 
                            names=names)
    result = {}
    if kwargs.get('return_gdf'):
        stat = pd.merge(stat, gdf[['id','geometry']], on='id')
        stat = gpd.GeoDataFrame(stat)
    
    result['df'] = stat

    if kwargs.get('return_all'):
        result |= msk

    return result
    
def get_data_agesex(
    gdf: Union[gpd.GeoDataFrame, str], 
    dataset: Optional[str] = 'R2025A',
    year: Optional[int] = 2020,
    resolution: Optional[str] ='100m',
    age_range: List[int]=[0,90],
    sex: Optional[str] = 'both',
    get_total: Optional[bool] = True,
    vrt_dir: Optional[str]='vrt', 
    **kwargs):
    '''
    Extracts population in regions defined in the gdf, segregated by
    age and sex.
    
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
        - dataset: Dataset to extract
        - year: Year of the data
        - resolution: Use '100m' or '1km'
        - age_range: Age range to be extracted [min, max]
        - sex: Use 'male', 'female', or 'both'
        - get_total: Boolean to get total population count.
        - vrt_dir: Directory of the virtual files defining the
                   raster mosaic.
        - **kwargs: Additional keyword arguments.

        Valid keyword arguments:
        - edge: Boolean to use the edge or boundary of Polygon for zonal
                aggregation. If true, the Polygon will be transformed into
                LineString.
        - rad: Buffer radius in km. Will be used particularly when
               the input vector contains Point or LineString.
        - weighted: Boolean to perform radial weight function over
                    the buffer area. This only apply for Point or
                    LineString geometry.
        - p: Exponent of the weight function:
             w(r) = (1 - np.exp(-(1-r)**p))/(1 - np.exp(-1)).
        - return_gdf: Boolean to return GeoDataFrame with geometry.
        - return_all: Boolean to return a dictionary containing
                        mask and weighting layer for zonal statistics
                        and also the resulting DataFrame. The keys of 
                        this object are 'df', 'mask', 'weight', 'kernel',
                        'bounds', 'transform'.
    Returns:
        DataFrame or GeoDataFrame with population count at age-sex
        structure. If return_all = True, the output includes some other 
        relevant intermediate products (mask, weight, kernel) and information
        (bounds, transform).

    Example:
        gdf = gpd.read_file('adm.gpkg')
        result = get_data_agesex(gdf, resolution='1km', age_range=[0,20],
            return_gdf=True)
        result.to_file('output.geojson', index=False)
        
    '''
    
    if resolution == '1km':
        folder = '1km_ua'
    else:
        folder = '100m'
    
    age_min = max(0, age_range[0])
    age_max = min(90, age_range[-1])
    age_groups = list(range(age_min, age_max+1, 5))
    if age_range[0] == 0:
        age_range.insert(1, 1)
        
    names = []
    if sex in ['female', 'both']:
        names += [f'f_{a:02d}' for a in age_groups]
    if sex in ['male', 'both']:
        names += [f'm_{a:02d}' for a in age_groups]
    
    vrt_paths = []
    for n in names:
        path = f'{vrt_dir}/{dataset}/agesex/{folder}/{year}/mosaic_{n}_{year}_{resolution}_constrained.vrt'
        vrt_paths.append(path)
    if get_total:
        path = f'{vrt_dir}/{dataset}/pop/{folder}/{year}/mosaic_{year}_{resolution}_constrained.vrt'
        vrt_paths.append(path)
        names.append('pop')

    result = extract(gdf, vrt_paths, resolution=resolution, names=names, **kwargs)
    if kwargs.get('return_all'):
        return result
    else:
        return result['df']
    
def get_data(
    gdf: Union[gpd.GeoDataFrame, str], 
    dataset: Optional[str] = 'R2025A',
    year: Optional[int] = 2020,
    resolution: Optional[str] = '100m',
    vrt_dir: Optional[str] = 'vrt', 
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
        - dataset: Dataset to extract
        - year: Year of the data
        - resolution: Use '100m' or '1km'
        - vrt_dir: Directory of the virtual files defining the
                   raster mosaic.
        - **kwargs: Additional keyword arguments.
                    
    Valid keyword arguments:
        - edge: Boolean to use the edge or boundary of Polygon for zonal
                aggregation. If true, the Polygon will be transformed into
                LineString.
        - rad: Buffer radius in km. Will be used particularly when
               the input vector contains Point or LineString.
        - weighted: Boolean to perform radial weight function over
                    the buffer area. This only apply for Point or
                    LineString geometry.
        - p: Exponent of the weight function:
             w(r) = (1 - np.exp(-(1-r)**p))/(1 - np.exp(-1)).
        - return_gdf: Boolean to return GeoDataFrame with geometry.
        - return_all: Boolean to return a dictionary containing
                        mask and weighting layer for zonal statistics
                        and also the resulting DataFrame. The keys of 
                        this object are 'df', 'mask', 'weight', 'kernel',
                        'bounds', 'transform'.
    Returns:
        DataFrame or GeoDataFrame with population count at age-sex
        structure. If return_all = True, the output includes some other 
        relevant intermediate products (mask, weight, kernel) and information
        (bounds, transform).
    
    Example:
        gdf = gpd.read_file('adm.gpkg')
        result = get_data_agesex(gdf, resolution='1km', age_range=[0,20],
            return_gdf=True)
        result.to_file('output.geojson', index=False)
        
    '''

    if resolution == '1km':
        folder = '1km_ua'
    else:
        folder = '100m'

    vrt_path = f'{vrt_dir}/{dataset}/pop/{folder}/{year}/mosaic_{year}_{resolution}_constrained.vrt'

    result = extract(gdf, vrt_paths, resolution=resolution, names=names, **kwargs)
    if kwargs.get('return_all'):
        return result
    else:
        return result['df']

if __name__ == '__main__':
    get_data()