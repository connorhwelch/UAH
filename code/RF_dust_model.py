# -*- coding: utf-8 -*-

"""Script to run Dust model with new RGB recipe

Script for both day,night,& night-enhanced dust RF model. Using Correct RGB Recipe


Usage
------

$ python rf_dust_model.py

Functions
----------

Dependencies
-------------

Notes
------
Code originally adapted from Nicholas Elmer (FEB 2020)

"""

__author__ = "Robert Junod"
__version__ = "0.2.0"
__license__ = "GNU GPLv3"
__update__ = "06/21/2022"
__pyVer__ = "Python 3.7"
__status__ = "Development"

from pathlib import Path
from datetime import datetime as dt
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import rasterio


def calc_split(barray,w1,w2):
    """
    Calculate split windows based on input
    """

    return barray[w1:w1+1:, ...] - barray[w2:w2+1, ...]


def calc_rgb(r, g, b):
    """
    Calculate rgb bands based on New NOAA RGB recipe

    Bands           ; Min to Max    ; Gamma 
    15 - 13 (red)   ; -6.7 to 2.6   ; 1.0 
    14 - 11 (green) ; -0.5 to 20.0  ; 2.5
    13      (blue)  ; 261.2 to 288.7; 1.0

    Parameters
    -----------
    r : ndarray
        "Red" band for RGB dust
    g : ndarray
        "Green" band for RGB dust
    b : ndarray
        "Blue" band for RGB dust
    """

    _b13 = b.copy()

    r[r > 2.6] = 2.6
    r[r < -6.7] = -6.7
    g[g > 20] = 20
    g[g < -0.5] = -0.5
    b[b > 288.7] = 288.7
    b[b < 261.2] = 261.2

    r = 255 * (r + 6.7) / (2.6 + 6.7)
    g = 255 * ((g + 0.5) / (20+0.5)) ** (1/2.5)
    b = 255 * (b - 261.2) / (288.7 - 261.2)

    r = r.astype('uint8')
    g = g.astype('uint8')
    b = b.astype('uint8')

    idx_b13 = _b13 < 90
    if np.count_nonzero(idx_b13) > 0:
        r[idx_b13] = 0
        g[idx_b13] = 0
        b[idx_b13] = 0

    return np.stack((r,g,b))


def calc_rgb_eu(r, g, b):
    """
    Calculate rgb bands based on Old EUMETSAT RGB recipe

    Bands           ; Min to Max    ; Gamma 
    15 - 14 (red)   ; -4 to 2       ; 1.0 
    14 - 11 (green) ; 0 to 15.0     ; 2.5
    13      (blue)  ; 261 to 289    ; 1.0

    Parameters
    -----------
    r : ndarray
        "Red" band for RGB dust
    g : ndarray
        "Green" band for RGB dust
    b : ndarray
        "Blue" band for RGB dust
    """

    _b13 = b.copy()

    r[r > 2] = 2
    r[r < -4] = -4
    g[g > 15] = 15
    g[g < 0] = 0
    b[b > 289] = 289
    b[b < 261] = 261

    r = 255 * (r + 4) / (2 + 4)
    g = 255 * ((g - 0) / (15-0)) ** (1/2.5)
    b = 255 * (b - 261) / (289 - 261)

    r = r.astype('uint8')
    g = g.astype('uint8')
    b = b.astype('uint8')

    idx_b13 = _b13 < 90
    if np.count_nonzero(idx_b13) > 0:
        r[idx_b13] = 0
        g[idx_b13] = 0
        b[idx_b13] = 0

    return np.stack((r,g,b))


def read_in(bfiles: list,cfiles: list,vfiles=[None], day_flag=False):
    """
    Read in training data from repository on sc1 
    
    Parameters
    -----------
    bfiles: list
        List of raster band files in .tif format
    cfiles: list
        List of labeled vector files in .tif format
    vfiles: list
        List of raster vis band files in .tif format
    day_flag: bool
        Day/Night flag 

    Returns
    --------
    ndarray
        Multi-dimesional stacked array of each "band" [band idx, values]

    ndarray
        "True" values for dust in 1D array

    """

    # Make sure number of files are equal
    if len(bfiles) != len(cfiles):
        raise ValueError(f'Number of files do no match! {bfiles} != {cfiles}')

    # If day_flag False, make
    if not day_flag:
        vfiles = [None]*len(bfiles)
    
    # Loop over input files
    for i, file in enumerate(zip(bfiles,cfiles,vfiles)):

        # band_data => IR Bands B10,B11,B13,B14,B15,B16
        band_data = rasterio.open(file[0])
        clip_data = rasterio.open(file[1])
        try:    
            vis_data = rasterio.open(file[2]) if day_flag else None
        except:
            vis_data = None
        
        if band_data.count == 6:
            barray = band_data.read()
        else:
            # We assume that there is 6 bands
            # there may be RTMA data if more than 6 bands
            # if not day_flag:
            #     print(f'{band_data.name}: {band_data.count} bands instead of 6')
            #     raise ValueError('Invalid number of bands')
            print(f'{band_data.name}: {band_data.count} bands instead of 6')
            barray = band_data.read()[:6,:,:]
        try:
            varray = vis_data.read() if day_flag else None
        except:
            varray = np.zeros_like(barray[:3,:,:])

        carray = clip_data.read()

        # Calculate split windows
        # Indices based on specific band order
        #   idx 4 => B15
        #   idx 3 => B14
        #   idx 2 => B13
        #   idx 1 => B11

        s_15_14 = calc_split(barray,4,3)
        s_15_13 = calc_split(barray,4,2)
        s_14_11 = calc_split(barray,3,1)

        if DFLAG:
            rgb_arr = calc_rgb(s_15_13.squeeze(),
                               s_14_11.squeeze(),
                               barray[2, :, :])
        else:
            rgb_arr = calc_rgb_eu(s_15_13.squeeze(),
                                  s_14_11.squeeze(),
                                  barray[2, :, :])



        # breakpoint()
        # combine initial bands and calculated "bands"
        if day_flag:
            all_bands = np.concatenate((barray,
                                        s_15_14,
                                        s_15_13,
                                        s_14_11,
                                        rgb_arr,
                                        varray))
        else:
            all_bands = np.concatenate((barray,
                                        s_15_14,
                                        s_15_13,
                                        s_14_11,
                                        rgb_arr))
            
        
        _x = all_bands.transpose(1, 2, 0).reshape(-1, all_bands.shape[0])
        classmask = carray.flatten()
        cidx = np.where((classmask == 0) | (classmask == 1))[0]
        
        _x = _x[cidx,:]
        _y = classmask[cidx].copy()

        if i != 0:
            _xall = np.concatenate((_xall,_x))
            _yall = np.concatenate((_yall,_y))
        else:
            _xall = _x.copy()
            _yall = _y.copy()

        print(f"{band_data.name.split('_')[-4]}: {i}, {len(_y)}, {np.unique(carray)}")

        del _x,_y, all_bands, rgb_arr, s_15_14, s_15_13, s_14_11
                

    return _xall, _yall


def rf_model(_trees, _min_samples, _x, _y):
    """
    Fit data to random forest model

    Parameters
    -----------
    _trees: int
        Number of trees to use
    _min_samples: int
        Minimum number of samples

    Return
    -------
    scikit learn model
        Random forest model

    """

    _clf = RandomForestClassifier(n_estimators = _trees,
                                  n_jobs = -1,
                                  min_samples_split = _min_samples*2,
                                  min_samples_leaf = _min_samples,
                                  random_state = 42
                                  )

    _clf.fit(_x,_y)
    
    return _clf


if __name__ == "__main__":
    # Directory of post-processed training/validation/testing data
    GEOTIFF_DIR = Path('/raid1/sport/people/rjunod/ML_DATA/DUST/')
    # Training data location for published NT-RF model
#     TRAIN_DUST_DIR = GEOTIFF_DIR / 'RF_20200820' / 'training'
    
    # Training data location for daytime model
    # TRAIN_DUST_DIR = GEOTIFF_DIR / 'RF_20210602' / 'training'

    # Training data location for NT-enhanced RF model
    # - Has daytime cases as well without visible bands
#     TRAIN_DUST_DIR = GEOTIFF_DIR / 'RF_20220802_night' / 'training'
    
    # Training data location for NT-enhanced RF model w/ visible bands
    # - Has daytime cases as well with visible bands
    # TRAIN_DUST_DIR = GEOTIFF_DIR / 'RF_20220802_night' / 'training'

    # Training data location for daytime RF model
    # - Has daytime cases only
    TRAIN_DUST_DIR = GEOTIFF_DIR / 'RF_20230327_day' / 'training'

    
    # Parameters for RF model
    NTREES = 200
    MIN_SAMPLES = 300

    # Flag for dust RGB recipe
    DFLAG = True  # True => New RGB; False => Old RGB EUMETSAT
    
    # Flag for day/night
    DAY_FLAG = True
    d_dict = {True:'day',False:'night'}
    
    B_FILES = sorted(list(TRAIN_DUST_DIR.glob('*all_bands_clip.tif')))
    C_FILES = sorted(list(TRAIN_DUST_DIR.glob('*raster_clip.tif')))
    
    if DAY_FLAG:
        V_FILES = sorted(list(TRAIN_DUST_DIR.glob('*vis_bands_clip.tif')))
        if len(B_FILES) != len(V_FILES):
            # create list with empty values
            _vfiles = np.empty(len(B_FILES), dtype=object)
            # Get date strings for V_FILES
            vdates = [v.name.split('_')[1] for v in V_FILES]
            VIDX = [i for i,b in enumerate(B_FILES) if b.name.split('_')[1] in vdates]
            _vfiles[VIDX] = V_FILES
            V_FILES = _vfiles.copy()
        
        XALL, YALL = read_in(B_FILES,C_FILES,V_FILES,DAY_FLAG)
    else:
        XALL, YALL = read_in(B_FILES,C_FILES)

    breakpoint()
    
    CLF = rf_model(NTREES, MIN_SAMPLES, XALL, YALL)

    M_NAME = f"GOES_dust_model_RF"

    DUST_STR = '' if DFLAG else '_EUMETSAT'
    # Save model to output directory
    # text file for list of scenes used for training
    # List of dates
    DATES = [i.name.split('_')[1] for i in B_FILES]
    SAVE_FILE = f'{M_NAME}_{dt.today():%Y%m%d}{DUST_STR}_{d_dict[DAY_FLAG]}'

    
    # Check path
    try:
        Path(f'../{M_NAME}').mkdir()
    except FileExistsError:
        print(f"{Path(f'../{M_NAME}')} exists")

    joblib.dump(CLF, f'../{M_NAME}/{SAVE_FILE}.sav', 3)
    np.savetxt(f'../{M_NAME}/{SAVE_FILE}.txt',
               DATES,
               header='Files',
               footer=f'{len(B_FILES)}',fmt='%s'
               )
    

    
    
    

        






