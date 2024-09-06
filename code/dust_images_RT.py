# -*- coding: utf-8 -*-
"""Program to create .png images from RF-DUST model

SEE README for details (add later)

INPUT: 

OUTPUT:

FUTURE: Add command line args later

NOTE: Needs improvement in dynamic scaling, sorta hard-coded

"""

__author__ = "Robert Junod"
__version__ = "0.2.0"
__license__ = "GNU GPLv3"
__update__ = "07/05/2022"
__pyVer__ = "Python 3.7"
__status__ = "Development"

# =============================================================================
#                                 IMPORT MODULES
# =============================================================================

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests

import dust_detect_eval as dust_eval
from pathlib import Path

from matplotlib import use
use('Agg')

# %matplotlib qt5

import matplotlib.pyplot as plt


def get_s3_keys(bucket, s3_client, prefix=''):
    kwargs = {'Bucket': bucket}
    
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix
    
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                yield key
            
        try:
            kwargs['ContinuationToken'] = resp['NextContiuationToken']
        except KeyError:
            break


def reproject_dust(p_rf, x, y, curr_crs, date):
    """ Temp function to reproject to different GeoTIFF"""
    import rioxarray as rxr
    import xarray as xr
    
    da = xr.DataArray(data=p_rf.reshape(y.size,x.size)*100,
                      dims=['y','x'],
                      coords={'y': y.values,
                              'x': x.values}
                     )
    # Only keep probs above 20
    da = da.where(da >= 20).copy()
    da.rio.write_crs(curr_crs.proj4_init, inplace=True)
    # reproject & write
    da_gcs = da.rio.reproject('EPSG:4326')
    da_gcs = da_gcs.rio.write_nodata(-9999)
    da_gcs.rio.to_raster(f'./dust_03_15_to_03_16_2022/dust_prob_{date:%Y_%m_%dT%H%M}_WGS84.tif')
    return 'SUCCESS!'


def create_images(file, pathin, model, resp, cextent):
    
    from datetime import datetime as dt
    import joblib
    import numpy as np
    import geopandas as gpd
    import cartopy.io.shapereader as shpreader

    
    def test_read(file, resp, model_type='night'):
        """ Test function to read in .nc files

        Test to see if script working correctly. Not needed in Real-Time

        Parameters
        ----------
        model_type: str, optional
            ML model type to be used; night or day (default is night)

        Returns
        --------
        crs object
            CRS object for mapping
        ndarray
            1D array for ML inputs
        ndarray
            1D array for ML "truth" values
        ndarray
            Stacked array of all bands needed for ML model

        Notes
        -----
        Needs metpy installed and xarray

        """ 

        # Packages for GOES 16 processing
        import xarray as xr
        import netCDF4
        import metpy

        # Check to make sure model_type is valid
        assert model_type.lower() in ['night','day'], 'Invalid model_type!'
        
        fname = file.split('/')[-1].split('.')[0]
        nc4_ds = netCDF4.Dataset(fname, memory = resp.content)
        store = xr.backends.NetCDF4DataStore(nc4_ds)
        data = xr.open_dataset(store)
        
        # Get crs and x, y coordinates info for later
        dat =  data.metpy.parse_cf('CMI_C02')
        x = dat.x
        y = dat.y
        goes_crs = dat.metpy.cartopy_crs

        # Subset Bands
        b10 = data.CMI_C10.copy()
        b11 = data.CMI_C11.copy()
        b13 = data.CMI_C13.copy()
        b14 = data.CMI_C14.copy()
        b15 = data.CMI_C15.copy()
        b16 = data.CMI_C16.copy()

        # Convert NAN to 0 in Bands

        b10 = xr.where(b10.isnull(), 0.0, b10)
        b11 = xr.where(b11.isnull(), 0.0, b11)
        b13 = xr.where(b13.isnull(), 0.0, b13)
        b14 = xr.where(b14.isnull(), 0.0, b14)
        b15 = xr.where(b15.isnull(), 0.0, b15)
        b16 = xr.where(b16.isnull(), 0.0, b16)

        # If ML day model
        if model_type.lower() == 'day':
            # Need to add in clip of data
            b1 = data.CMI_C01.copy()
            b2 = data.CMI_C02.copy()
            _b3 = data.CMI_C03.copy()
            
            b3 = 0.48358168 * b2 + 0.45706946 * b1 + 0.06038137 * _b3

            b1 = b1.clip(0,1).copy()
            b2 = b2.clip(0,1).copy()
            b3 = b3.clip(0,1).copy()
            
            b1 = xr.where(b1.isnull(), 0.0, b1)
            b2 = xr.where(b2.isnull(), 0.0, b2)
            b3 = xr.where(b3.isnull(), 0.0, b3)
        
        # Calculate Split bands
        b_1514 = b15.values - b14.values 
        b_1411 = b14.values - b11.values
        b_1513 = b15.values - b13.values
        
        #######################################################################
        #                          RGB recipe NOAA
        #######################################################################
        # Bands;     Min to Max; Gamma 
        # 15 - 13;  -6.7 to 2.6; 1.0 
        # 14 - 11;  -0.5 to 20.0; 2.5
        # 13;      261.2 to 288.7; 1.0
        
        _r = b_1513.copy()
        _g = b_1411.copy()
        _b = b13.data.copy()
        
        _r[_r > 2.6] = 2.6
        _r[_r < -6.7] = -6.7
        _g[_g > 20] = 20
        _g[_g < -0.5] = -0.5
        _b[_b > 288.7] = 288.7
        _b[_b < 261.2] = 261.2
        
        r = 255 * (_r + 6.7) / (2.6 + 6.7)
        g = 255 * ((_g + 0.5) / (20+0.5)) ** (1/2.5)
        b = 255 * (_b - 261.2) / (288.7 - 261.2)
        
        #######################################################################
        #                         RGB recipe EUMETSAT
        #######################################################################
#         # Calculate RGB
        # _r = b_1514.copy()
        # _g = b_1411.copy()
        # _b = b13.data.copy()
        # # From GOESR_process_dust.py code
        # # RGB limits
        # _r[_r > 2] = 2
        # _r[_r < -4] = -4
        # _g[_g > 15] = 15
        # _g[_g < 0] = 0
        # _b[_b > 289] = 289
        # _b[_b < 261] = 261

        # r = 255 * (_r + 4.) / 6.
        # g = 255 * (_g / 15.) ** (1/2.5)
        # b = 255 * (_b - 261) / (289 - 261)
        #######################################################################
        
        
        r = r.astype('uint8')
        g = g.astype('uint8')
        b = b.astype('uint8')

        # Some B13 check
        idx_b13 = b13.data < 90

        if b13.where(idx_b13).count() > 0:
            r[idx_b13] = 0
            g[idx_b13] = 0
            b[idx_b13] = 0

        if model_type.lower() == 'day':
            return goes_crs, x, y, np.stack((b10, b11, b13, b14, b15, b16,
                                             b_1514, b_1513, b_1411,
                                             r, g, b,
                                             b1, b2, b3), axis=-1), data
        else:
            return goes_crs, x, y, np.stack((b10, b11, b13, b14, b15, b16,
                                             b_1514, b_1513, b_1411,
                                             r, g, b), axis=-1), data
    def plot_goes_separate(p_rf, rgb, date, goes_crs, extent='', x='', y='', dust_cmap='jet_r', path_out = './', extent_conus= ''):
        """Plot separate RGB and dust

        Plot Dust RGB image and probabilities onto 2 panel plot

        Parameters
        ----------
        p_rf: ndarray
            1D array of probabilities
        rgb: ndarray
            stacked array of "Red","Green", & "Blue bands
        date: datetime object
            Requested date for plot
        goes_crs: crs
            CRS for the GOES ABI grid
        extent: tuple, optional
            tuple of extent (default is '')
        x: ndarray, optional
            x-coordinates of image (default is '')
        y: ndarray, optional
            y-coordinates of image (default is '')
        dust_cmap: LinearSegmentColormap, optional
            cmap to use for dust (default is 'jet_r')

        Returns
        -------
        str
            Success or Failure string

        Notes
        -----
        Either "extent" or "x/y" need to be defined for this function to work
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

#         try:
        #  default map extent
        if extent_conus == '':
            extent_conus = [-126.,-66.,24., 50.]

        if extent == '':
            assert (len(x) != 0) & (len(y) != 0), "Need to define extent or x/y parameters"
            extent = (x.min(),x.max(),y.min(),y.max())
        
        # figure size calculations
        fig_height = extent_conus[3] - extent_conus[2]
        fig_width = extent_conus[1] - extent_conus[0]

        fig_size = ((fig_width/fig_height)*4, 4)
        
        # font size dynamic sizing
#         fsize12 = 12*(fig_width/fig_height)
#         fsize10 = 10*(fig_width/fig_height)
        
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': goes_crs})
        ax.set_title(f"Random Forest Dust Probabilities {date:%Y-%m-%d %H:%M} UTC", fontsize='xx-small')
        p_rf[p_rf < .20] = np.nan
        cs = ax.imshow(p_rf.reshape(rgb.shape[0],rgb.shape[1])*100, origin='upper',
                       cmap=dust_cmap, clim=(20,100), 
                       transform=goes_crs,zorder=10,
                       extent=extent)
        ax.set_extent(extent_conus, crs= ccrs.Geodetic())
        ax.coastlines(color='white', linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5,facecolor='black', edgecolor='white',zorder=8)
        ax.add_feature(cfeature.STATES, linewidth=0.5,edgecolor='white',zorder=11)
        # With geopandas
        ax.add_geometries(UScounties.geometry, ccrs.PlateCarree(), edgecolor='white',facecolor='none', linewidth=0.25, zorder=11)
        # Without geopandas
#         ax.add_geometries(UScounties.geometries(), ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=0.5, zorder=9)
        

#         fig.subplots_adjust(top=0.9, bottom=0.0,left=0.0, right=0.88)
        ax.set_aspect('auto')
        # fig.tight_layout(rect=(0,0,0.88,1), pad=0.0)
        cbar_ax = fig.add_axes(cbar_coords)
        cbar = fig.colorbar(cs, cax=cbar_ax)
        cbar.set_label('Dust Probability',fontsize=4)
        cbar.ax.tick_params(labelsize=4, length=2)
        ax._set_position(fig_rect)
        plt.savefig(path_out / 'dust_prob' / f'dust_prob_{date:%Y_%m_%dT%H%M}_{MODEL_FLAG}_NE.png', dpi=DPI)
        plt.close()
        
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': goes_crs})
        ax.imshow(rgb, origin='upper', transform=goes_crs, extent=extent)
        ax.set_extent(extent_conus, crs= ccrs.Geodetic())
        ax.coastlines(color='black', linewidth=0.5,zorder=11)
        ax.add_feature(cfeature.STATES, linewidth=0.5,zorder=11)
        # With geopandas
        ax.add_geometries(UScounties.geometry, ccrs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.25, zorder=11)
        # Without geopandas
#         ax.add_geometries(UScounties.geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, zorder=9)
        
        ax.set_title(f"GOES-16 ABI Dust RGB {date:%Y-%m-%d %H:%M} UTC", fontsize='xx-small')
        ax.set_aspect('auto')
        # fig.tight_layout(rect=(0,0,0.88,1), pad=0.0)
        ax._set_position(fig_rect)
        plt.savefig(path_out / 'RGB' / f'dust_rgb_{date:%Y_%m_%dT%H%M}_{MODEL_FLAG}_NE.png', dpi=DPI)
        plt.close()
#         except:
#             return 'Fail!'

        return 'Success!'
    def plot_goes_ml_overlay(p_rf, rgb, date, goes_crs, extent='', x='', y='', dust_cmap='jet_r', path_out = './', extent_conus = ''):
        """Plot overlay

        Plot that overlays Dust probability contours onto RGB image

        Parameters
        ---------,zorder=11-
        p_rf: ndarra,zorder=11y
            1D array of probabilities
        rgb: ndarray
            stacked array of "Red","Green", & "Blue bands
        date: datetime object
            Requested date for plot
        goes_crs: crs
            CRS for the GOES ABI grid
        extent: tuple, optional
            tuple of extent (default is '')
        x: ndarray, optional 
            x-coordinates of image (default is '')
        y: ndarray, optional
            y-coordinates of image (default is '')
        dust_cmap: LinearSegmentColormap, optional
            cmap to use for dust (default is 'jet_r')

        Returns
        -------
        str
        String of Success or Failure

        Notes
        -----
        Either "extent" or "x/y" need to be defined for this function to work

        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib import colors

        if extent_conus == '':
            extent_conus = [-126.,-66.,24., 50.]

        if extent == '':
            assert (len(x) != 0) & (len(y) != 0), "Extent or x/y parameters not defined"
            extent = (x.values.min(),x.values.max(),y.values.min(),y.values.max())
        
        # figure size calculations
        fig_height = extent_conus[3] - extent_conus[2]
        fig_width = extent_conus[1] - extent_conus[0]

        fig_size = ((fig_width/fig_height)*4, 4)
        
        # font size dynamic sizing
#         fsize12 = 12*(fig_width/fig_height)
#         fsize10 = 10*(fig_width/fig_height)
        
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': goes_crs})

        # Plot DUST RGB
        ax.imshow(rgb, origin='upper', transform=goes_crs,
                extent=extent)
        ax.set_extent(extent_conus, crs= ccrs.Geodetic())
        ax.coastlines(color='black', linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=11)
        
        # With geopandas
        ax.add_geometries(UScounties.geometry, ccrs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.25, zorder=11)
        # Without geopandas
#         ax.add_geometries(UScounties.geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, zorder=9)
        
        ax.set_title(f"GOES-16 ABI Dust RGB with Dust Probabilities {date:%Y-%m-%d %H:%M} UTC",fontsize='xx-small')
        # Overlay Dust Probabilities
        p_rf[p_rf < .20] = np.nan
        cs = ax.imshow(p_rf.reshape(rgb.shape[0],rgb.shape[1])*100, origin='upper',
                    cmap=dust_cmap, clim=(20,100), 
                    transform=goes_crs,zorder=10,
#                     alpha=0.5,
                    extent=extent)

        #     cs = ax.contour(p_rf.reshape(rgb.shape[0],rgb.shape[1])*100,
        #                     origin='upper', extent=extent, cmap=dust_cmap,
        #                     levels=np.arange(20,101,10), 
        #                     transform=goes_crs)
        #     norm = colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        #     sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        #     sm.set_array([])

        # New plotting code
        ax.set_aspect('auto')
        # print(ax.get_position())
        cbar_ax = fig.add_axes(cbar_coords)
        #     cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar = fig.colorbar(cs, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=4, length=2)
        cbar.set_label('Dust Probability', fontsize=4)
        ax._set_position(fig_rect)

        plt.savefig(path_out / f'overlay_dust_{date:%Y_%m_%dT%H%M}_{MODEL_FLAG}_NE.png', dpi=DPI)    
        plt.close()

        return 'Success'
    
    def combine_overlay_plots(prob, rgb, date, goes_crs, extent='',
                              x='', y='', dust_cmap='jet_r',
                              path_out = './', extent_conus = '', panel=2):
        """Combine NT and DT """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib import colors

        if extent_conus == '':
            extent_conus = [-126.,-66.,24., 50.]

        if extent == '':
            assert (len(x) != 0) & (len(y) != 0), "Extent or x/y parameters not defined"
            extent = (x.values.min(),x.values.max(),y.values.min(),y.values.max())
        
        # figure size calculations
        fig_height = extent_conus[3] - extent_conus[2]
        fig_width = extent_conus[1] - extent_conus[0]

        fig_size = ((fig_width/fig_height)*(panel*4), 4)
        
        # font size dynamic sizing
#         fsize12 = 12*(fig_width/fig_height)
#         fsize10 = 10*(fig_width/fig_height)
        
        fig, ax = plt.subplots(figsize=fig_size, ncols=panel, subplot_kw={'projection': goes_crs})

#         model_str = {2:['Day', 'Night'], 3:['Day', 'Night', 'Night+Day']}
        model_str = {2:['Day', 'Night'], 3:['Night', 'Night+Day w/o vis', 'Night+Day w/ vis']}
    
        for axes,_p_rf,_model in zip(ax,prob,model_str[panel]):
            # Plot DUST RGB
            axes.imshow(rgb, origin='upper', transform=goes_crs,
                    extent=extent)
            axes.set_extent(extent_conus, crs= ccrs.Geodetic())
            axes.coastlines(color='black', linewidth=0.5)
            axes.add_feature(cfeature.STATES, linewidth=0.5, zorder=11)

            # With geopandas
            axes.add_geometries(UScounties.geometry, ccrs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.25, zorder=11)

            # Overlay Dust Probabilities
            axes.set_title(f"Random Forest {_model} Model",fontsize='xx-small')
            
            cs = axes.imshow(_p_rf.reshape(rgb.shape[0],rgb.shape[1])*100,
                             origin='upper', cmap=dust_cmap, clim=(20,100), 
                             transform=goes_crs,zorder=10,
                             extent=extent)

            #     cs = ax.contour(p_rf.reshape(rgb.shape[0],rgb.shape[1])*100,
            #                     origin='upper', extent=extent, cmap=dust_cmap,
            #                     levels=np.arange(20,101,10), 
            #                     transform=goes_crs)
            #     norm = colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
            #     sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
            #     sm.set_array([])

            # New plotting code
            axes.set_aspect('auto')
            # print(ax.get_position())
            
            # ax._set_position(fig_rect)
        fig.suptitle(f"GOES-16 ABI Dust RGB with Dust Probabilities {date:%Y-%m-%d %H:%M} UTC",fontsize='x-small')
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        cbar = fig.colorbar(cs, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=4, length=2)
        cbar.set_label('Dust Probability', fontsize=4)
        
        out_str = {2:"both",3:"three"}
        plt.savefig(path_out / f'overlay_dust_{date:%Y_%m_%dT%H%M}_{out_str[panel]}.png', dpi=DPI, bbox_inches = 'tight')    
        plt.close()
        return 'Success'
        
    DATE = dt.strptime(file.split('_')[-3][1:-3], '%Y%j%H%M')
    GOES_CRS,X,Y,ALL_BANDS,DATA = test_read(file, resp, MODEL_FLAG)

    if 'day' in MODEL_FLAG:
        RF_MODEL_DT = joblib.load(pathin / model)
        RF_MODEL_NT = joblib.load(pathin / 'GOES_dust_model_RF_20220802_night.sav')
        RF_MODEL_NT2 = joblib.load(pathin / 'GOES_dust_model_RF_20220809_day.sav')
        # Assuming visible bands in ALL_BANDS
        P_RF_DT = RF_MODEL_DT.predict_proba(ALL_BANDS[:,:,:12].reshape(-1,ALL_BANDS[:,:,:12].shape[-1]))[:, 1]
        P_RF_NT = RF_MODEL_NT.predict_proba(ALL_BANDS[:,:,:12].reshape(-1,ALL_BANDS[:,:,:12].shape[-1]))[:, 1]
        P_RF_NT2 = RF_MODEL_NT2.predict_proba(ALL_BANDS.reshape(-1,ALL_BANDS.shape[-1]))[:, 1]

        P_RF_DT[P_RF_DT < .2] = np.nan
        P_RF_NT[P_RF_NT < .2] = np.nan
        P_RF_NT2[P_RF_NT2 < .2] = np.nan

        P_RF = P_RF_DT.copy()
    else:
        RF_MODEL = joblib.load(pathin / model)
        P_RF = RF_MODEL.predict_proba(ALL_BANDS.reshape(-1,ALL_BANDS.shape[-1]))[:, 1]
    
    # With geopandas
    UScounties = gpd.read_file('./UScounties/UScounties.shp')
    # Without geopandas
#     UScounties = shpreader.Reader('./UScounties/UScounties.shp')

    # Get RGB stacked array from ALL_BANDS
    if MODEL_FLAG == 'night':
        RGB = np.dstack((ALL_BANDS[...,-3],
                         ALL_BANDS[...,-2],
                         ALL_BANDS[...,-1])).astype('uint8')
    else:
        RGB = np.dstack((ALL_BANDS[...,-6],
                         ALL_BANDS[...,-5],
                         ALL_BANDS[...,-4])).astype('uint8')

    # ====================================================================
    #                               Plotting
    # ====================================================================
    # Custom cmap
    CMAP_COLORS = [(255,255,255),(255,255,0),(255,165,0),(255,0,0),
                   (255,0,255),(139,0,139)]
    DUST_CMAP = dust_eval.make_cmap(CMAP_COLORS, bit=True)
    
    
    # Only works when 'day' is initially used
    # if 'day' in MODEL_FLAG:
    #     # print(f'Plotting both overlay plot {DATE}....',end='')
    #     # STATUS = combine_overlay_plots([P_RF_DT,P_RF_NT,P_RF_NT2], RGB, DATE, GOES_CRS,
    #     #                                x=X, y=Y, dust_cmap = DUST_CMAP,
    #     #                                path_out = (PATH_OUT/f'{DATE:%Y%m%d}'/'overlay'),
    #     #                                extent_conus = cextent, panel=3)
    #     # STATUS = combine_overlay_plots([P_RF_DT,P_RF_NT], RGB, DATE, GOES_CRS,
    #     #                                x=X, y=Y, dust_cmap = DUST_CMAP,
    #     #                                path_out = (PATH_OUT/f'{DATE:%Y%m%d}'/'overlay'),
    #     #                                extent_conus = cextent, panel=2)
    #     print(STATUS)
    #     plt.close()
    #     plt.clf()
        
    print(f'Plotting overlay plot {DATE}....',end='')
    STATUS = plot_goes_ml_overlay(P_RF, RGB, DATE, GOES_CRS, x=X, y=Y, dust_cmap = DUST_CMAP,
                                  path_out = (PATH_OUT/f'{DATE:%Y%m%d}'/'overlay'),
                                  extent_conus = cextent)
    print(STATUS,end='...')
# # #     print(STATUS)
    plt.close()
# #     plt.clf()
    print(f'Plotting separate plot {DATE}....',end='')
    STATUS = plot_goes_separate(P_RF, RGB, DATE, GOES_CRS,
                                x=X, y=Y, dust_cmap = DUST_CMAP,
                                path_out = (PATH_OUT/f'{DATE:%Y%m%d}'),
                                extent_conus = cextent)
    print(STATUS)
    plt.close()
    plt.clf()
#     return GOES_CRS,X,Y,ALL_BANDS,P_RF,DATA,DATE
    return GOES_CRS,X,Y,P_RF,DATE


if __name__ == "__main__":

    B_NAME = 'noaa-goes16'
    P_NAME = 'ABI-L2-MCMIPC'
    YR = 2023
    DOY = 121
    HOUR = [15,16,17,18,19] # Adjust as needed
    # HOUR = [22,23] # Adjust as needed
    MODE = 'M6' # M6 for anything after April 2nd, 2019 M3 for before
    MODEL_FLAG = 'night'
    DPI = 200
    
    # OLD MODELS
    # RF_NAME = 'GOES_dust_model_RF_20200820.sav' if (MODEL_FLAG == 'night') else 'GOES_dust_daymodel_RF_20210602.sav'
    # NEW MODELS w/ updated RGB Recipe
    
#     RF_NAME = 'GOES_dust_model_RF_20220519_night.sav' if (MODEL_FLAG == 'night') \
#          else 'GOES_dust_model_RF_20220713_day.sav'
    
    RF_NAME = 'GOES_dust_model_RF_20220802_night.sav'
    
    # Location of model code    
    PATH_IN = Path('/usr/people/rjunod/DUST_ML/source_files/GOES_dust_model_RF')
    # Location of output images
    DIR_OUT = Path('/raid1/sport/people/rjunod/ML_DATA/DUST/')
    PATH_OUT = DIR_OUT / f'RF_{RF_NAME.split("_")[-2]}_{MODEL_FLAG}' / 'cases_png'
    # PATH_OUT = DIR_OUT / f'RF_{RF_NAME.split("_")[-2]}_night' / 'cases_png'

    fig_rect = (0.05, 0.05, 0.83, 0.83)
    cbar_coords = [0.89, 0.3, 0.02, 0.4]

    # March 7th 2018
    # cextent = [-103.74, -91.13, 31.98, 38.65]

    # March 16th 2018
    # cextent = [-109.24, -101.28, 28.60, 35.70]

    # April 17th 2019
    # cextent = [-120.5, -111.76, 32.31, 38.51]

    # March 17th 2021
    # cextent = [-108.22, -96.95, 28.86, 36.33]

    # SW domain full
    # cextent = [-124.,-90., 24., 45.]

    # SW domain subset
    # CEXTENT = [-109, -95.6, 27.1, 40.8]

    # Custom Domain
    CEXTENT = [-93.61, -86.02, 36.67, 43.0]

    # Test 
    # cextent = [-109, -103.86, 30.9, 34.76]

    # cextent = ''



    S3_CLIENT = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    for HR in HOUR:
        KEYS = get_s3_keys(B_NAME, S3_CLIENT,
                        prefix=f'{P_NAME}/{YR}/{DOY:03}/{HR:02}/OR_{P_NAME}-{MODE}')
        KEY = [KEY for KEY in KEYS]
        for FILE in KEY:
            RESP = requests.get(f'https://{B_NAME}.s3.amazonaws.com/{FILE}')
    #         GOES_CRS,X,Y,ALL_BANDS,P_RF,DATA,DATE = create_images(file,PATH_IN,RF_NAME, resp, cextent)
            GOES_CRS,X,Y,P_RF,DATE = create_images(FILE,PATH_IN,RF_NAME, RESP, CEXTENT)
    #         _status = reproject_dust(P_RF, X, Y, GOES_CRS, DATE)
            print(f'{DATE:%Y_%m_%dT%H%M} DONE!')
        #     break
        # break
    print('DONE!!!')
