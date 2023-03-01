import pylab as pl
from astropy.coordinates import SkyCoord 

from utils import *
from grm_utils import *

verbose = True

# -----------------------------------------------------------
# step 1: read catalogues
xmm12_file = '/Users/ascaife/DATA/MIGHTEE/catalog/XMMLSS/XMMLSS_12/XMMLSS_12_1538856059_1539286252_pol_detections.fits'
xmm12c = SkyCoord(34.4625,-4.83306,unit="deg")
xmm12 = read_catalogue(xmm12_file, xmm12c)

xmm13_file = '/Users/ascaife/DATA/MIGHTEE/catalog/XMMLSS/XMMLSS_13/XMMLSS_13_1538942495_1539372679_pol_detections.fits'
xmm13c = SkyCoord(35.175,-4.83306,unit="deg")
xmm13 = read_catalogue(xmm13_file, xmm13c)

xmm14_file = '/Users/ascaife/DATA/MIGHTEE/catalog/XMMLSS/XMMLSS_14/XMMLSS_14_1539028868_1539460932_pol_detections.fits'
xmm14c = SkyCoord(35.84167,-4.83306,unit="deg")
xmm14 = read_catalogue(xmm14_file, xmm14c)

cosmos_file = '/Users/ascaife/DATA/MIGHTEE/catalog/COSMOS/COSMOS_1587911796_1524147354_1525613583_pol_detections.fits'
cosmosc = SkyCoord(35.84167,-4.83306,unit="deg") # update this
cosmos = read_catalogue(cosmos_file, cosmosc)

# -----------------------------------------------------------
# step 2: merge catalogues
if verbose: print("--- \n >>> Merging catalogues")

mightee = merge_cats([xmm12, xmm13, xmm14, cosmos])
print('# of sources in merged catalogue: {}'.format(mightee.shape[0]))

# -----------------------------------------------------------
# step 2a: Russ' filter to remove XMMLSS filament:
if verbose: print("--- \n >>> Filtering XMMLSS filament")

polflux_limit = 70.0; frac_p_limit = 30.0
idx = mightee[(mightee['pol'] < polflux_limit) & (mightee['P/I'] > frac_p_limit)].index
mightee = mightee.drop(index = idx)
print('# of sources in filtered catalogue: {}'.format(mightee.shape[0]))

# -----------------------------------------------------------
# step 2b: match spectroscopic redshifts from Catherine/Imogen
if verbose: print("--- \n >>> Matching z_spec")

cosmos_zcat = '/Users/ascaife/SRC/GITHUB/MIGHTEE-Bfield/redshifts/COSMOSXMATCH+multiinfo_v5.fits'
cosmosc = SkyCoord(35.84167,-4.83306,unit="deg") # update this
cosmos_z = read_catalogue(cosmos_zcat, cosmosc, cols=['RADIORA','RADIODEC'])

mightee = match_z(cosmos_z, mightee)

# -----------------------------------------------------------
# step 3: redshift and galactic latitude filtering
if verbose: print("--- \n >>> Filtering redshift and gal lat")

mightee = filter_z(mightee, has_z=True, spec_z=False)
print('# of sources (after z filter): {}'.format(mightee.shape[0]))

# redshift histogram:
pl.rcParams["figure.figsize"] = (8,8)
pl.subplot(111)
n, bins, patches = pl.hist(mightee['best_z'], density=False, bins=20, edgecolor='w', facecolor='b', alpha=0.75)
pl.xlabel('Redshift ($z$)')
pl.ylabel('Frequency')
pl.savefig('plots/mightee_z.png')
#pl.show()

# add columns for (l,b)
mightee['l'], mightee['b'] = j2000_to_gal(mightee['ra'].values, mightee['dec'].values)

# filter b<25:
mightee = filter_glat(mightee, bmin=25.)
print('# of sources (after b filter): {}'.format(mightee.shape[0]))

# -----------------------------------------------------------
# step 4: get GRMs
if verbose: print("--- \n >>> Getting GRM values")

ra = np.array(mightee['ra'].values)
dec= np.array(mightee['dec'].values)

fitsfile = './faradaysky/faraday2020v2.fits'
grmdata, grmhp = get_grm_data(fitsfile) # from grm_utils.py

# nearest pixel values:
grm_fs, grm_fs_std = get_grm(ra, dec, grmdata, grmhp) # from grm_utils.py
mightee['GRM_FS'] = grm_fs
mightee['GRM_FSerr'] = grm_fs_std

# 1 degree median values:
grm1_med, grm1_mad, grm1_std = get_grm1(ra, dec, grmdata, grmhp, r=0.5) # 1 degree median values (grm_utils)
mightee['GRM1'] = grm1_med
mightee['GRM1err'] = grm1_mad

pl.rcParams["figure.figsize"] = (8,5)
pl.errorbar(grm_fs, grm1_med, xerr=grm_fs_std, yerr=grm1_mad, ls='', marker='o')
pl.plot([-10,10],[-10,10], ls=':')
pl.xlabel('GRM [direct]')
pl.ylabel('GRM [1 degree median]')
pl.savefig('./plots/grm.png')
#pl.show()

# -----------------------------------------------------------
# step 5: subtract off GRMs
if verbose: print("--- \n >>> Subtracting GRMs")

rm      = mightee['RM'].values
rm_err  = mightee['RM_err'].values
grm     = mightee['GRM1'].values      # use 1 deg median
grm_err = mightee['GRM1err'].values   # use 1 deg median
z       = mightee['best_z'].values

# take difference
rrm = rm - grm

print("---")
print("Using GRM1 subtraction: ")
rrm_rms = np.sqrt(np.std(rrm) - np.std(rm_err) - np.std(grm_err))
se1 = standard_error_on_stdev(np.std(rrm), len(rrm))
se2 = standard_error_on_stdev(np.std(rm_err), len(rm_err))
se3 = standard_error_on_stdev(np.std(grm_err), len(grm_err))
rrm_rms_ste = np.sqrt(se1**2 + se2**2 + se3**2)
print("<RRM^2>^0.5 : {:.2f} +/- {:.2f} rad/m^2".format(rrm_rms, rrm_rms_ste))


# remove data outside 2 sigma:
std = np.std(rrm)
grm_err = grm_err[np.where(np.abs(rrm)<=2*std)]
rrm = rrm[np.where(np.abs(rrm)<=2*std)]
print('# of sources (after 2sig rm subtraction): {}'.format(len(rrm)))

rrm_rms_obs = stats.median_abs_deviation(rrm, scale='normal', nan_policy='omit')
rrm_mad = stats.median_abs_deviation(rrm, nan_policy='omit')

print("---")
print("Using GRM subtraction: ")
print("<RRM^2>^0.5 [direct] : {:.2f} rad/m^2".format(np.std(rrm)))
print("RRM MAD : {:.2f} rad/m^2".format(rrm_mad))
print("<RRM^2>^0.5 [using MAD] : {:.2f} rad/m^2".format(rrm_rms_obs))
print("Average GRM error : {:.2f} rad/m^2".format(np.mean(grm_err)))

pl.rcParams["figure.figsize"] = (8,8)
pl.subplot(111)
n, bins, patches = pl.hist(rrm, density=False, range=(-25,25), bins=50, edgecolor='w', facecolor='orange', alpha=1.0)
pl.xlabel('RRM')
pl.ylabel('Frequency')
pl.savefig('./plots/mightee_rrms.png')
#pl.show()

# -----------------------------------------------------------
# step 6: calculate f_excess
if verbose: print("--- \n >>> Calculating f_excess")

rm      = mightee['RM'].values
rm_err  = mightee['RM_err'].values
grm     = mightee['GRM_FS'].values      # use 1 deg median
grm_err = mightee['GRM_FSerr'].values   # use 1 deg median
grm1    = mightee['GRM1'].values        # use 1 deg median
z       = mightee['best_z'].values

eps = 1e-7
rm_excess = rm  - grm1
grm_excess= grm - grm1 + eps

f_excess = rm_excess/grm_excess

print("---")
print("fexcess: {:.2f} +/- {:.2f} ".format(np.median(f_excess), stats.median_abs_deviation(f_excess, nan_policy='omit')))
print("fexcess stdev: {:.2f}".format(stats.median_abs_deviation(f_excess, scale='normal', nan_policy='omit')))

#pl.rcParams["figure.figsize"] = (8,5)
#pl.scatter(np.arange(len(f_excess)), f_excess, s=1)
#pl.xlabel('# source')
#pl.ylabel('fexcess')
#pl.ylim(-3,3)
#pl.show()
#pl.savefig('./plots/mightee_fexcess.png')

pl.rcParams["figure.figsize"] = (8,8)
pl.subplot(111)
n, bins, patches = pl.hist(f_excess, density=False, range=(-50,50), bins=20, edgecolor='w', facecolor='orange', alpha=1.0)
pl.xlabel(r'$f_{excess}$')
pl.ylabel('Frequency')
pl.savefig('./plots/mightee_fexcess.png')
#pl.show()

# -----------------------------------------------------------
# step 7: 
if verbose: print("--- \n >>> Making redshift bins")

rm      = mightee['RM'].values
rm_err  = mightee['RM_err'].values
grm     = mightee['GRM_FS'].values      # use 1 deg median
grm_err = mightee['GRM_FSerr'].values   # use 1 deg median
grm1    = mightee['GRM1'].values        # use 1 deg median
z       = mightee['best_z'].values

rrm = rm - grm1

std = np.std(rrm)
rrm = rrm[np.where(np.abs(rrm)<=2*std)]
z = z[np.where(np.abs(rrm)<=2*std)]

bins, idx_bin, n_bin = bin_z(z, nbins=7)
mu_bin = 0.5*np.diff(bins)+bins[:-1]

#print("Number of sources per bin:",n_bin)
#print("Central redshift per bin:",mu_bin)

mu_rrm = [np.average(rrm[idx_bin == j]) for j in range(1, len(bins))]
sig_rrm= [np.std(rrm[idx_bin == j]) for j in range(1, len(bins))]
ste_rrm= sig_rrm/np.sqrt(n_bin)
ste_sig = standard_error_on_stdev(sig_rrm, n_bin)

#print("Means: {}".format(mu_rrm))
#print("Standard errors:", ste_rrm)

pl.rcParams["figure.figsize"] = (8,5)
pl.scatter(z, rrm, s=1)
pl.errorbar(mu_bin, mu_rrm, yerr=ste_rrm, xerr=0.5*np.diff(bins), c='red', ls='', capsize=3.)
pl.xlabel('z')
pl.ylabel('RRM [rad/m^2]')
pl.ylim(-40,40)
pl.savefig('./plots/mightee_rrm_z.png')
#pl.show()

pl.scatter(z, np.abs(rrm), s=1)
pl.errorbar(mu_bin, sig_rrm, yerr=ste_sig, xerr=0.5*np.diff(bins), c='red', ls='', capsize=3.)
pl.xlabel('z')
pl.ylabel('RRM rms [rad/m^2]')
pl.ylim(0,40)
pl.savefig('./plots/mightee_rrmsig_z.png')
#pl.show()
