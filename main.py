from utils import *
from grm_utils import *

# step 1: read catalogues
xmm12_file = '/Users/ascaife/DATA/MIGHTEE/catalog/XMMLSS/XMMLSS_12/XMMLSS_12_1538856059_1539286252_pol_detections.fits'
xmm12 = read_catalogue(xmm12_file)
xmm13_file = '/Users/ascaife/DATA/MIGHTEE/catalog/XMMLSS/XMMLSS_13/XMMLSS_13_1538942495_1539372679_pol_detections.fits'
xmm13 = read_catalogue(xmm13_file)
xmm14_file = '/Users/ascaife/DATA/MIGHTEE/catalog/XMMLSS/XMMLSS_14/XMMLSS_14_1539028868_1539460932_pol_detections.fits'
xmm14 = read_catalogue(xmm14_file)
cosmos_file = '/Users/ascaife/DATA/MIGHTEE/catalog/COSMOS/COSMOS_1587911796_1524147354_1525613583_pol_detections.fits'
cosmos = read_catalogue(cosmos_file)

# step 2: merge catalogues
mightee = merge_cats([xmm12, xmm13, xmm14, cosmos])

# step 3: redshift and galactic latitude filtering

# step 4: get GRMs

# step 5: subtract off GRMs

# step 6: calculate f_excess

# step 7: 