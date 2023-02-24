from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Quantity
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
import astropy.units as u

import numpy as np
import scipy.stats as stats


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def get_grm_data(grmfile):

    grmhdu  = fits.open(grmfile)
    grmdata = (grmhdu[1].data["faraday_sky_mean"],grmhdu[1].data["faraday_sky_std"])
    hp = HEALPix(nside=512, order='ring', frame=Galactic())
    
    return grmdata, hp

# -------------------------------------------------------------------------------

def get_grm(ra, dec, data, hp):

    """
    Function to return value of GRM map at specific coordinate (nearest pixel)
    """
    
    # for an array of values:
    if hasattr(ra, "__len__"):
        grm_mean = np.zeros_like(ra)
        grm_err  = np.zeros_like(ra)
        
        nsrc = len(ra)
        for i in range(nsrc):
            coord = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg, frame=FK5)
            hp_idx  = hp.skycoord_to_healpix(coord)
    
            grm_mean[i] = data[0][hp_idx]
            grm_err[i]  = data[1][hp_idx]
    
    # for a single value:
    else:
    
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=FK5)
        hp_idx  = hp.skycoord_to_healpix(coord)
        
        grm_mean = data[0][hp_idx]
        grm_err  = data[1][hp_idx]
            
    return grm_mean, grm_err
    
# -------------------------------------------------------------------------------

def get_grm1(ra, dec, data, hp, r=0.5):
        
    """
    Function to median of GRM map over 1 degree diameter centred on coordinate
    """

    if hasattr(ra, "__len__"):
    
        grm_med = np.zeros_like(ra)
        grm_mad = np.zeros_like(ra)
        grm_std = np.zeros_like(ra)
    
        nsrc = len(ra)
        for i in range(nsrc):
            
            pos    = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg, frame=FK5)
            pixels = hp.cone_search_skycoord(pos, radius=r * u.deg)
    
            grm_mean = data[0][pixels]
            grm_err  = data[1][pixels]

            grm_med[i] = np.median(grm_mean)
            grm_mad[i] = stats.median_abs_deviation(grm_mean)
            grm_std[i] = np.std(grm_mean)
    else:
        pos    = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=FK5)
        pixels = hp.cone_search_skycoord(pos, radius=r * u.deg)
        
        grm_mean = data[0][pixels]
        grm_err  = data[1][pixels]

        grm_med = np.median(grm_mean)
        grm_mad = stats.median_abs_deviation(grm_mean)
        grm_std = np.std(grm_mean)
        
    return grm_med, grm_mad, grm_std

# -------------------------------------------------------------------------------

def get_grm_lb(ra, dec, data, hp):

    """
    Function to return value of GRM map at specific coordinate (nearest pixel)
    """
    
    # for an array of values:
    if hasattr(ra, "__len__"):
        grm_mean = np.zeros_like(ra)
        grm_err  = np.zeros_like(ra)
        
        nsrc = len(ra)
        for i in range(nsrc):
            coord = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg, frame=FK5)
            coord = coord.galactic
            hp_idx  = hp.lonlat_to_healpix(coord.l.deg, coord.b.deg, return_offsets=False)
            grm_mean[i] = data[0][hp_idx]
            grm_err[i]  = data[1][hp_idx]
    
    # for a single value:
    else:
    
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=FK5)
        coord = coord.galactic
        hp_idx  = hp.lonlat_to_healpix(coord.l, coord.b, return_offsets=False)
        grm_mean = data[0][hp_idx]
        grm_err  = data[1][hp_idx]
            
    return grm_mean, grm_err
    
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
