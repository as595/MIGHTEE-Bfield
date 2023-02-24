from astropy.io import fits
from astropy.table import Table
import pandas as pd


def read_catalogue(catfile):

    #with fits.open(catfile) as data:
    #    df = pd.DataFrame(data[0].data)
    
    dat = Table.read(catfile, format='fits')
    df = dat.to_pandas()
    
    return df


def merge_cats(catlist):

    df = pd.concat(catlist)
        
    return df

    