from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
import pandas as pd
import numpy as np
import warnings

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def read_catalogue(catfile, field_centre):

    warnings.filterwarnings('ignore', category=AstropyWarning)
    dat = Table.read(catfile, format='fits')

    field = catfile.split('/')[-2]
    dat.add_column(field,name='pointing')

    coord = SkyCoord(dat['ra'],dat['dec'],unit="deg")
    distr = coord.separation(field_centre).degree
    dat.add_column(distr,name='distance')
    
    df = dat.to_pandas()
    
    return df

# -------------------------------------------------------------------------------
# updated from Russ' version:

def merge_cats(catlist, distol=10.):

    ncat = len(catlist)
    nmatches = 0
    for c1 in range(0,ncat-1):
        for c2 in range(c1+1,ncat):
            j=0
            while True:
                found,n,dra,ddec = a_in_b(catlist[c1].iloc[[j]],catlist[c2],distol)
                if found:
                    nmatches = nmatches + 1
                    if catlist[c1].iloc[[j]]['distance'].values[0] < catlist[c2].iloc[[n]]['distance'].values[0]:
                        catlist[c2] = catlist[c2].drop(index=n).reset_index(drop=True)
                        j+=1
                    else:
                        catlist[c1] = catlist[c1].drop(index=j).reset_index(drop=True)
                else:
                    j+=1

                if j>=len(catlist[c1]['ra'].values): break

    print("Found {} matches".format(nmatches))
    df = pd.concat(catlist)
            
    return df

# -------------------------------------------------------------------------------
# from Russ:

def a_in_b(a,b,distol):
    found = False
    dis = []; dras = []; ddecs = []
    for i in range(len(b['ra'].values)):   
        dd = 3600.0*(b['dec'].values[i] - a['dec'].values)
        dr = 3600.0*(b['ra'].values[i] - a['ra'].values)*np.cos(a['dec'].values*np.pi/180.0)
        dis.append(np.sqrt(dd*dd + dr*dr))
        dras.append(dr)
        ddecs.append(dd)
    dis = np.array(dis)
    n = np.argmin(dis)
    if dis[n] < distol:
        found = True
    return found, n, dras[n], ddecs[n]

# -------------------------------------------------------------------------------

def filter_z(df, has_z=True, spec_z=True, zmax=2.):

    if has_z:
        df = df.loc[df['best_z'] != -99]
    
    if spec_z:
        df = df.loc[df['spec_z'] != -99]
    
    if zmax!=None:
        df = df.loc[df['best_z'] < zmax]
    
    return df
    
# -------------------------------------------------------------------------------

def filter_glat(df, bmin=25.):

    df = df.loc[df['b'].abs() > bmin]

    return df

# -------------------------------------------------------------------------------

def j2000_to_gal(ra, dec):

    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='fk5')
    coord = coord.transform_to('galactic') 
    
    return coord.l.degree, coord.b.degree


# -------------------------------------------------------------------------------

def standard_error_on_stdev(std, n):

    """
    Function to return the standard error on the sample
    standard deviation assuming a Normal distribution
    https://math.stackexchange.com/questions/1259383/calculating-uncertainty-in-standard-deviation
    https://stats.stackexchange.com/questions/156518/what-is-the-standard-error-of-the-sample-standard-deviation
    """
    
    if hasattr(n, "__len__"):
        std = np.array(std)
        n   = np.array(n)
    
    se = std/np.sqrt(2*n)
    
    return se

# -------------------------------------------------------------------------------

def standard_error_on_variance(std, n):

    """
    Function to return the standard error on the sample
    standard deviation assuming a Normal distribution
    https://math.stackexchange.com/questions/1259383/calculating-uncertainty-in-standard-deviation
    https://stats.stackexchange.com/questions/156518/what-is-the-standard-error-of-the-sample-standard-deviation
    """
    
    if hasattr(n, "__len__"):
        std = np.array(std)
        n   = np.array(n)
    
    se = np.sqrt((2*std**4)/(n-1))
    
    return se

# -------------------------------------------------------------------------------

def bin_z(z, nbins=7):

    d_z = 2./nbins
    bins = np.arange(0,(nbins+1)*d_z,d_z)
    mu_bin = 0.5*np.diff(bins)+bins[:-1]
    idx_bin = np.digitize(z, bins)

    n_bin = [len(z[idx_bin==j]) for j in range(1, len(bins))]

    return bins, idx_bin, n_bin

# -------------------------------------------------------------------------------
