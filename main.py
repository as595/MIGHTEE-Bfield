import pylab as pl
from astropy.coordinates import SkyCoord 
from scipy.optimize import minimize
from scipy import stats

from utils import *
from grm_utils import *

# User inputs:
# -----------------------------------------------------------
# -----------------------------------------------------------

verbose = True
bintype = 'width'  # ['number', 'width']
grmtype = 'grm1'    # ['grm1', 'direct]

# -----------------------------------------------------------
# -----------------------------------------------------------


# Main code [no changes below here]:
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

mightee_pre_z = mightee.copy(deep=True)

cosmos_zcat = '/Users/ascaife/SRC/GITHUB/MIGHTEE-Bfield/redshifts/COSMOSXMATCH+multiinfo_v5.fits'
cosmosc = SkyCoord(35.84167,-4.83306,unit="deg") # update this
cosmos_z = read_catalogue(cosmos_zcat, cosmosc, cols=['RADIORA','RADIODEC'])

mightee = match_z(cosmos_z, mightee)

# -----------------------------------------------------------
# step 3: redshift and galactic latitude filtering
if verbose: print("--- \n >>> Filtering by redshift")

mightee = filter_z(mightee, has_z=True, spec_z=False)
print('# of sources (after z filter): {}'.format(mightee.shape[0]))

nspec = mightee.loc[mightee['spec_z'] != -99].shape[0]
if verbose: print('# of spectroscopic redshifts: {}'.format(nspec))
if verbose: print('# of photometric redshifts: {}'.format(mightee.shape[0]-nspec))


# redshift histogram:
pl.rcParams["figure.figsize"] = (8,8)
pl.subplot(111)
n, bins, patches = pl.hist(mightee['best_z'], density=False, bins=20, edgecolor='w', facecolor='b', alpha=0.5)
pl.xlabel('Redshift ($z$)')
pl.ylabel('Frequency')
pl.savefig('plots/mightee_z.png')
pl.close()
#pl.show()

if verbose: print("--- \n >>> Filtering by Galactic lat")
# add columns for (l,b)
mightee['l'], mightee['b'] = j2000_to_gal(mightee['ra'].values, mightee['dec'].values)

# filter b<25:
mightee = filter_glat(mightee, bmin=25.)
print('# of sources (after b filter): {}'.format(mightee.shape[0]))
if verbose: print('minimum latitude: {} deg'.format(np.min(np.abs(mightee['b'].values))))

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
#pl.errorbar(grm_fs, grm1_med, xerr=grm_fs_std, yerr=grm1_mad, ls='', marker='o')
pl.scatter(grm_fs, grm1_med, ls='', marker='o')
pl.plot([-10,15],[-10,15], ls=':')
pl.xlabel('GRM [direct]')
pl.ylabel('GRM [1 degree median]')
pl.savefig('./plots/grm.png')
pl.axis([-5,12,-5,12])
pl.close()
#pl.show()

# -----------------------------------------------------------
# step 5: subtract off GRMs
if verbose: print("--- \n >>> Subtracting GRMs")

rm      = mightee['RM'].values
rm_err  = mightee['RM_err'].values
z       = mightee['best_z'].values

if grmtype=='grm1':
    grm     = mightee['GRM1'].values      # use 1 deg median
    grm_err = mightee['GRM1err'].values   # use 1 deg median
elif grmtype=='direct':
    grm     = mightee['GRM_FS'].values      # use 1 deg median
    grm_err = mightee['GRM_FSerr'].values   # use 1 deg median


# take difference
rrm = rm - grm

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

# -----------------------------------------------------------
# step 5a: subtract off GRMs
if verbose: print("--- \n >>> Subtracting GRMs")

rm      = mightee['RM'].values
rm_err  = mightee['RM_err'].values
z       = mightee['best_z'].values

if grmtype=='grm1':
    grm     = mightee['GRM1'].values      # use 1 deg median
    grm_err = mightee['GRM1err'].values   # use 1 deg median
elif grmtype=='direct':
    grm     = mightee['GRM_FS'].values      # use 1 deg median
    grm_err = mightee['GRM_FSerr'].values   # use 1 deg median

# take difference
rrm = rm - grm
rrm_err = np.sqrt(rm_err**2 + grm_err**2)

std = np.std(rrm)
grm_err = grm_err[np.where(np.abs(rrm)<=2*std)]
rm_err = rm_err[np.where(np.abs(rrm)<=2*std)]
rrm_err = rrm_err[np.where(np.abs(rrm)<=2*std)]
z = z[np.where(np.abs(rrm)<=2*std)]
rrm = rrm[np.where(np.abs(rrm)<=2*std)]

# calculate rrm rms subtracting rm errors and grm errors:
rrm_rms = np.sqrt(np.std(rrm)**2 - np.std(rm_err)**2 - np.std(grm_err)**2)
se1 = standard_error_on_variance(np.std(rrm), len(rrm))
se2 = standard_error_on_variance(np.std(rm_err), len(rm_err))
se3 = standard_error_on_variance(np.std(grm_err), len(grm_err))
ste = np.sqrt((1/(2*rrm_rms))**2*(se1**2+se2**2+se3**2))
print("Using GRM1 subtraction: ")
print("<RRM^2>^0.5 : {:.2f} +/- {:.2f} rad/m^2".format(rrm_rms, ste))

pl.rcParams["figure.figsize"] = (8,8)
pl.subplot(111)
n, bins, patches = pl.hist(rrm, density=False, range=(-25,25), bins=50, edgecolor='w', facecolor='orange', alpha=1.0)
pl.xlabel('RRM')
pl.ylabel('Frequency')
pl.savefig('./plots/mightee_rrms.png')
pl.close()
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

pl.rcParams["figure.figsize"] = (8,5)
pl.scatter(np.arange(len(f_excess)), f_excess, s=1)
pl.xlabel('# source')
pl.ylabel('fexcess')
pl.ylim(-50,50)
#pl.show()
pl.savefig('./plots/mightee_fexcess1.png')
pl.close()

pl.rcParams["figure.figsize"] = (8,8)
pl.subplot(111)
n, bins, patches = pl.hist(f_excess, density=False, range=(-50,50), bins=20, edgecolor='w', facecolor='orange', alpha=1.0)
pl.xlabel(r'$f_{excess}$')
pl.ylabel('Frequency')
pl.savefig('./plots/mightee_fexcess2.png')
pl.close()
#pl.show()

# -----------------------------------------------------------
# step 7: 
if verbose: print("--- \n >>> Making redshift bins")

rm      = mightee['RM'].values
rm_err  = mightee['RM_err'].values
z       = mightee['best_z'].values

if grmtype=='grm1':
    grm     = mightee['GRM1'].values      # use 1 deg median
    grm_err = mightee['GRM1err'].values   # use 1 deg median
elif grmtype=='direct':
    grm     = mightee['GRM_FS'].values      # use 1 deg median
    grm_err = mightee['GRM_FSerr'].values   # use 1 deg median

rrm = rm - grm
rrm_err = np.sqrt(rm_err**2 + grm_err**2)

# remove data outside 2 sigma:
std = np.std(rrm)
grm_err = grm_err[np.where(np.abs(rrm)<=2*std)]
rm_err = rm_err[np.where(np.abs(rrm)<=2*std)]
rrm_err = rrm_err[np.where(np.abs(rrm)<=2*std)]
z = z[np.where(np.abs(rrm)<=2*std)]
rrm = rrm[np.where(np.abs(rrm)<=2*std)]

# make redshift bins:
bins, idx_bin, n_bin = bin_z(z, nbins=7, bintype=bintype)
mu_bin = 0.5*np.diff(bins)+bins[:-1]

# -----------------------------------------------------------------------------
# get the mean RRM in each bin:
mu_rrm = np.array([np.average(rrm[idx_bin == j]) for j in range(1, len(bins))])
# get the uncorrected rms RRM in each bin:
sig_rrm_obs= np.array([np.std(rrm[idx_bin == j]) for j in range(1, len(bins))])
# get the standard error on the mean for each bin:
ste_rrm= sig_rrm_obs/np.sqrt(n_bin)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# get the rms RM error in each bin: 
sig_rm_err = np.array([np.std(rm_err[idx_bin == j]) for j in range(1, len(bins))])
# get the rms GRM error in each bin:
sig_grm_err= np.array([np.std(grm_err[idx_bin == j]) for j in range(1, len(bins))])

# get the error corrected RRM rms in each bin:
sig_rrm = np.sqrt(sig_rrm_obs**2 - sig_rm_err**2 - sig_grm_err**2)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# calculate standard error on each rms:
se1 = standard_error_on_variance(sig_rrm_obs, n_bin)
se2 = standard_error_on_variance(sig_rm_err, n_bin)
se3 = standard_error_on_variance(sig_grm_err, n_bin)

# calculate the overall standard error on the corrected RRM rms:
ste_sig = np.sqrt((1/(2*sig_rrm))**2*(se1**2+se2**2+se3**2))
# -----------------------------------------------------------------------------

pl.rcParams["figure.figsize"] = (20,5)
pl.subplot(121)
pl.scatter(z, rrm, s=1, c='black')
pl.errorbar(mu_bin, mu_rrm, yerr=ste_rrm, xerr=0.5*np.diff(bins), c='red', ls='', capsize=3.)
pl.xlabel('z')
pl.ylabel(r'RRM [rad m$^{-2}$')
pl.ylim(-40,40)
#pl.title("Figure 4 (top)")


pl.subplot(122)
pl.scatter(z, np.abs(rrm), s=1, c='black')
pl.errorbar(mu_bin, sig_rrm, yerr=ste_sig, xerr=0.5*np.diff(bins), c='red', ls='', capsize=3.)
pl.xlabel('z')
pl.ylabel(r'RRM rms [rad m$^{-2}$')
pl.ylim(0,40)
#pl.title("Figure 4 (bottom)")

pl.savefig("./plots/mightee_rrm_z.png")
pl.close()

# create table:
d = {'z': mu_bin, 'n_src': n_bin, 'RRM_mu': mu_rrm, 'RRM_mu_err': ste_rrm, 'RRM_rms': sig_rrm, 'RRM_rms_err': ste_sig}
df = pd.DataFrame(data=d)
df = df.style.format(precision=2)
if verbose: print(df.to_latex())

# fit trend line to mean RRM:
np.random.seed(42)
nll = lambda *args: -log_likelihood_mbf(*args)
initial = np.array([0., 10., 1.]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(mu_bin, mu_rrm, ste_rrm))
m_ml, b_ml, log_f_ml = soln.x
ih = soln.hess_inv

if verbose: 
    print("Maximum likelihood estimates of fit to mean RRM:")
    print("m = {:.3f}+/-{:.3f}".format(m_ml, np.sqrt(ih[0,0])))
    print("b = {:.3f}+/-{:.3f}".format(b_ml, np.sqrt(ih[1,1])))
    print("f = {:.3f}+/-{:.3f}".format(np.exp(log_f_ml), np.sqrt(ih[2,2])))

# fit trend line to rms RRM:
soln = minimize(nll, initial, args=(mu_bin, sig_rrm, ste_sig))
m_ml, b_ml, log_f_ml = soln.x
ih = soln.hess_inv

if verbose: 
    print("Maximum likelihood estimates of fit to rms RRM:")
    print("m = {:.3f}+/-{:.3f}".format(m_ml, np.sqrt(ih[0,0])))
    print("b = {:.3f}+/-{:.3f}".format(b_ml, np.sqrt(ih[1,1])))
    print("f = {:.3f}+/-{:.3f}".format(np.exp(log_f_ml), np.sqrt(ih[2,2])))

# -----------------------------------------------------------------------------
# step 8a:
if verbose: print("--- \n >>> Applying redshift evolution models")

df = pd.DataFrame(mu_bin, columns=["z"])
df["n_src"] = n_bin
df["RRM mean"] = mu_rrm
df["RRM ste"] = ste_rrm
df["RRM rms"] = sig_rrm
df["RRM sig"] = ste_sig

rrm0 = zdep(rrm, z, model='C1')
df["C1 rms"], df["C1 sig"] = bin_rrm(rrm0, bins, idx_bin, n_bin)

rrm0 = zdep(rrm, z, model='C2')
df["C2 rms"], df["C2 sig"] = bin_rrm(rrm0, bins, idx_bin, n_bin)

rrm0 = zdep(rrm, z, model='C3')
df["C3 rms"], df["C3 sig"] = bin_rrm(rrm0, bins, idx_bin, n_bin)

pl.rcParams["figure.figsize"] = (5,5)
pl.scatter(z, np.abs(rrm), c='black', s=1)
pl.errorbar(mu_bin, sig_rrm, yerr=ste_sig, xerr=0.5*np.diff(bins), c='red', ls='-', capsize=3., label='RRM')
pl.errorbar(mu_bin, df["C1 rms"].values, yerr=df["C1 sig"].values, xerr=0.5*np.diff(bins), c='blue', ls='dotted', capsize=3., label='C1')
pl.errorbar(mu_bin, df["C2 rms"].values, yerr=df["C2 sig"].values, xerr=0.5*np.diff(bins), c='orange', ls='dashed', capsize=3., label='C2')
pl.errorbar(mu_bin, df["C3 rms"].values, yerr=df["C3 sig"].values, xerr=0.5*np.diff(bins), c='green', ls='dashdot', capsize=3., label='C3')
pl.xlabel('z')
pl.ylabel('RRM rms [rad/m^2]')
pl.ylim(0,80)
pl.legend()
pl.savefig('./plots/mightee_cmods.png')
pl.close()

# -----------------------------------------------------------------------------
# step 8b: 
if verbose: print("--- \n >>> Fitting redshift behaviour")

np.random.seed(42)
nll = lambda *args: -log_likelihood_mbf(*args)
initial = np.array([0., 10., 1.]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(mu_bin, df["C1 rms"], df["C1 sig"]))
m_ml, b_ml, log_f_ml = soln.x
ih = soln.hess_inv

if verbose: 
    print("Maximum likelihood estimates [C1]:")
    print("m = {:.3f}+/-{:.3f}   ({:.3f})".format(m_ml, np.sqrt(ih[0,0]), m_ml/np.sqrt(ih[0,0])))
    print("b = {:.3f}+/-{:.3f}".format(b_ml, np.sqrt(ih[1,1])))
    print("f = {:.3f}+/-{:.3f}".format(np.exp(log_f_ml), np.sqrt(ih[2,2])))
    print("Spearman rank coeff: {:.3f}".format(stats.spearmanr(mu_bin, df["C1 rms"]).statistic))
    
    print(" C1 & {:.2f}$\pm${:.2f} & {:.2f}$\pm${:.2f} & {:.2f} & {:.2f}".format(m_ml, np.sqrt(ih[0,0]), b_ml, np.sqrt(ih[1,1]), m_ml/np.sqrt(ih[0,0]), stats.spearmanr(mu_bin, df["C1 rms"]).statistic))

soln = minimize(nll, initial, args=(mu_bin, df["C2 rms"], df["C2 sig"]))
m_ml, b_ml, log_f_ml = soln.x
ih = soln.hess_inv

if verbose: 
    print("Maximum likelihood estimates [C2]:")
    print("m = {:.3f}+/-{:.3f}   ({:.3f})".format(m_ml, np.sqrt(ih[0,0]), m_ml/np.sqrt(ih[0,0])))
    print("b = {:.3f}+/-{:.3f}".format(b_ml, np.sqrt(ih[1,1])))
    print("f = {:.3f}+/-{:.3f}".format(np.exp(log_f_ml), np.sqrt(ih[2,2])))
    print("Spearman rank coeff: {:.3f}".format(stats.spearmanr(mu_bin, df["C2 rms"]).statistic))

    print(" C2 & {:.2f}$\pm${:.2f} & {:.2f}$\pm${:.2f} & {:.2f} & {:.2f}".format(m_ml, np.sqrt(ih[0,0]), b_ml, np.sqrt(ih[1,1]), m_ml/np.sqrt(ih[0,0]), stats.spearmanr(mu_bin, df["C2 rms"]).statistic))

soln = minimize(nll, initial, args=(mu_bin, df["C3 rms"], df["C3 sig"]))
m_ml, b_ml, log_f_ml = soln.x
ih = soln.hess_inv

if verbose: 
    print("Maximum likelihood estimates [C3]:")
    print("m = {:.3f}+/-{:.3f}   ({:.3f})".format(m_ml, np.sqrt(ih[0,0]), m_ml/np.sqrt(ih[0,0])))
    print("b = {:.3f}+/-{:.3f}".format(b_ml, np.sqrt(ih[1,1])))
    print("f = {:.3f}+/-{:.3f}".format(np.exp(log_f_ml), np.sqrt(ih[2,2])))
    print("Spearman rank coeff: {:.3f}".format(stats.spearmanr(mu_bin, df["C3 rms"]).statistic))

    print(" C3 & {:.2f}$\pm${:.2f} & {:.2f}$\pm${:.2f} & {:.2f} & {:.2f}".format(m_ml, np.sqrt(ih[0,0]), b_ml, np.sqrt(ih[1,1]), m_ml/np.sqrt(ih[0,0]), stats.spearmanr(mu_bin, df["C3 rms"]).statistic))
    sig_rrm_c3 = df["C3 rms"].values; ste_sig_c3 = df["C3 sig"].values
    
# -----------------------------------------------------------------------------

print(">--------<")
df["RRM mean"] = df["RRM mean"].round(2).astype(str)+"$\pm$"+df["RRM ste"].round(2).astype(str)
df["RRM rms"] = df["RRM rms"].round(2).astype(str)+"$\pm$"+df["RRM sig"].round(2).astype(str)
df["C1 rms"] = df["C1 rms"].round(2).astype(str)+"$\pm$"+df["C1 sig"].round(2).astype(str)
df["C2 rms"] = df["C2 rms"].round(2).astype(str)+"$\pm$"+df["C2 sig"].round(2).astype(str)
df["C3 rms"] = df["C3 rms"].round(2).astype(str)+"$\pm$"+df["C3 sig"].round(2).astype(str)
df = df.drop(columns='RRM ste')
df = df.drop(columns='RRM sig')
df = df.drop(columns='C1 sig')
df = df.drop(columns='C2 sig')
df = df.drop(columns='C3 sig')
df = df.style.format(precision=2)
if verbose: print(df.to_latex())

# -----------------------------------------------------------------------------
# step 9: 
if verbose: print("--- \n >>> Filament analysis")

np.random.seed(42)
nll = lambda *args: -log_likelihood_fil(*args)
initial = np.array([0., 1., 0.]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(mu_bin, sig_rrm_c3, ste_sig_c3))
m_ml, b_ml, log_f_ml = soln.x
ih = soln.hess_inv

print("Maximum likelihood estimates:")
print("m = {:.2f}+/-{:.2f}".format(m_ml, np.sqrt(ih[0,0])))
print("b = {:.2f}+/-{:.2f}".format(b_ml, np.sqrt(ih[1,1])))
print("f = {:.2f}+/-{:.2f}".format(np.exp(log_f_ml), np.sqrt(ih[2,2])))

model = m_ml*N_f(mu_bin)**0.5+b_ml
pl.errorbar(mu_bin, sig_rrm_c3, ste_sig_c3, xerr=0.5*np.diff(bins), c='green', ls='dashdot', capsize=3.)
pl.plot(mu_bin, model, c='orange')
pl.xlabel('z')
pl.ylabel('RRM0 (C3) rms [rad/m^2]')
pl.ylim(0,35)
pl.savefig('./plots/mightee_filament.png')
pl.close()

# -----------------------------------------------------------------------------
# b-field calculation
ne = 1e-5
ell = 3e6*(np.pi/2)
B_para = m_ml/(0.81*ne*ell)
B_para_err = np.sqrt(ih[0,0])/(0.81*ne*ell)
print("Magnetic field strength: {:.2f}+/-{:.2f} nG".format(B_para*1e3, B_para_err*1e3))