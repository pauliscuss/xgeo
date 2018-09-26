import requests
from os.path import join
from urllib.parse import urljoin
import rasterio
import pandas as pd
import numpy as np
import xarray as xr
import xgeo

def download_file(url, outdir):
    fn = url.split('/')[-1]
    path = join(outdir, fn)
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return path

# NOTE this function has not yet been tested
# NOTE opendap access seems this is not working with libnetcdf versions after v4.4.1.1
# NOTE the fileServer links for met_forcing_v1 don't work
def get_e2o_data(forcing, start_date, end_date, like, opendap=True, chunks={'lat': 100, 'lon': 100}):
    serverroot = u'http://wci.earth2observe.eu/thredds/'
    serverroot = urljoin(serverroot, 'dodsC/') if opendap else urljoin(serverroot, 'fileServer/')
    forcings = dict(
        Rainf = {'folder': u'jrc/MSWEP/daily_e2o_netcdf_convention/', 'fn':'Rainf_daily_MSWEP_025_{year}{month}.nc'},
        Snowf = {'folder': u'jrc/MSWEP/daily_e2o_netcdf_convention/', 'fn':'Rainf_daily_MSWEP_025_{year}{month}.nc'},
        Tair = {'folder': u'ecmwf/met_forcing_v1/', 'fn':'tair_daily.nc'},
        Psurf = {'folder': u'ecmwf/met_forcing_v1/', 'fn':'psurf_daily.nc'},
        Wind = {'folder': u'ecmwf/met_forcing_v1/', 'fn':'wind_daily.nc'},
        SWdown = {'folder': u'ecmwf/met_forcing_v1/', 'fn':'swdown_daily.nc'},
        LWdown = {'folder': u'ecmwf/met_forcing_v1/', 'fn':'lwdown_daily.nc'},
        PET = {'folder': u'deltares/PET/wrr2/0.25degree/', 'fn':'hargreaves-wrr2-agg.nc'}
        )
    base = urljoin(serverroot, forcings[forcing]['folder'])
    fn = forcings[forcing]['fn']
    if forcing in ['Rainf', 'Snowf']:
        urls = [urljoin(base, fn.format(year='{:04d}'.format(t.year), month='{:02d}'.format(t.month))) 
            for t in pd.date_range(start_date, end_date, freq='MS')]
    else:
        urls = [urljoin(base, fn)]

    # get output grid
    crs, transform, height, width = xgeo.utils.rasterio_grid(like)
    # NOTE this assumes that the like crs is also epsg:4326. 
    # TODO warp bounds to epsg:4326
    bounds = rasterio.transform.array_bounds(height, width, transform)

    # read input
    da = xr.open_mfdataset(urls, chunks=chunks)[forcing].sortby('time')
    # select / clip
    da = da.sel(**{'time': slice(start_date, end_date)})
    da = xgeo.regrid.clip(da, bounds)
    da.load()
    # set metadata
    da = xgeo.utils.set_crs(da, crs=rasterio.crs.CRS.from_epsg(4326))
    if forcing == 'Rainf':
        da = da * 86400
        da.attrs.update(units= 'mm')
    # reproject 
    reproject_kwargs = dict(like=like, resampling='average')
    da2 = xgeo.regrid.reproject_rasterio(da, **reproject_kwargs)
    return da2