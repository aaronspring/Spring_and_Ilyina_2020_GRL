import warnings
from collections import OrderedDict
from copy import copy

import cartopy.crs as ccrs
import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import xarray as xr
from climpred.bootstrap import (DPP_threshold, bootstrap_perfect_model,
                                varweighted_mean_period_threshold)
from climpred.prediction import compute_perfect_model, compute_persistence
from climpred.stats import DPP, autocorr, rm_trend, varweighted_mean_period
from esm_analysis.composite import composite_analysis
from esm_analysis.stats import linregress, rm_trend
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from PMMPIESM.plot import my_plot
from xskillscore import pearson_r

from basics import (_get_path, labels, longname, metric_dict, path_paper,
                    post_global, post_ML, shortname, units)

warnings.filterwarnings("ignore")
%matplotlib inline

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'

r_ppmw2ppmv = 28.8 / 44.0095
CO2_to_C = 44.0095 / 12.0111

savefig = False
savefig = True


# Fig SI ACF
acf = []
for i in range(11):
    acf.append(autocorr(control_global, lag=i))
acf = xr.concat(acf, 'time')
acf.to_dataframe().plot()
plt.ylabel('ACF [ ]')
plt.xlabel('Lag [year]')
plt.title('Autocorrelation function')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_ACF')


# composite_analysis

# Load ENSO
control3d = xr.open_dataset(_get_path(
    'tos', prefix='control'))
control3d = control3d - \
    control3d.rolling(time=10, center=True, min_periods=1).mean()

nino34_mask = xr.open_dataset(
    PM.setup.file_origin + 'masks/enso_34_mask.nc')['alon'].squeeze()
# nino34_mask.plot()

enso = (control3d * nino34_mask).mean(['x', 'y']).squeeze()['tos']
# enso.plot()
del enso['depth']

threshold = 1
# quiver plot
v = 'u10'
u10 = xr.open_dataset(_get_path(
    v, prefix='control'))[v].squeeze()
ano = (u10 - u10.mean('time'))
comp_u10 = composite_analysis(
    ano, enso, plot=False, ttest=True, robust=True, yincrease=True, cmap='coolwarm', threshold=threshold)

v = 'v10'
v10 = xr.open_dataset(_get_path(
    v, prefix='control'))[v].squeeze()
ano = (v10 - v10.mean('time'))
comp_v10 = composite_analysis(
    ano, enso, plot=False, ttest=True, robust=True, yincrease=True, cmap='coolwarm', threshold=threshold)

v = 'CO2'
CO2 = xr.open_dataset(_get_path(
    v, prefix='control'))[v].squeeze()
del CO2['lev']
CO2.name = 'CO$_2$ [ppm]'
ano = (CO2 - CO2.mean('time'))
comp_CO2 = composite_analysis(
    ano, enso, plot=False, robust=True, ttest=True, yincrease=True, cmap='coolwarm', threshold=threshold)

v = 'co2_flux'
co2_flux = xr.open_dataset(_get_path(
    v, prefix='control'))[v].squeeze()
ano = (co2_flux - co2_flux.mean('time'))
comp_co2_flux = composite_analysis(
    ano, enso, plot=False, robust=True, ttest=True, yincrease=True, cmap='coolwarm', threshold=threshold)


def quiver(background_map, u, v, proj=ccrs.PlateCarree(), ax=None, quiverreference_size=1, coastline_color='gray', qkx=0.85, qky=.95, **kwargs):
    """Plot arrows over background map."""
    if ax is None:
        ax = plt.axes(projection=proj)
    if background_map is None:
        ax.stock_img()
    else:
        background_map.plot.pcolormesh(
            'lon', 'lat', ax=ax, transform=ccrs.PlateCarree(), **kwargs)
    lon = u.lon.values
    lat = u.lat.values
    Q = ax.quiver(lon, lat, u.values, v.values, transform=ccrs.PlateCarree(
        central_longitude=-180), width=0.002, regrid_shape=25, color='black')
    ax.coastlines(color=coastline_color, linewidth=1.5)
    qk = plt.quiverkey(Q, qkx, qky, quiverreference_size, str(quiverreference_size) + r'$\frac{m}{s}$', labelpos='E',
                       coordinates='figure')


levels = 21
fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                         'projection': ccrs.PlateCarree()}, figsize=(15, 7))
quiver(CO2.mean('time'), u10.mean('time'), v10.mean('time'),
       quiverreference_size=10, levels=levels, robust=True, cmap='viridis', ax=axes[0, 0], qkx=.35, qky=.97)

axes[0, 0].set_title('Mean surface atmospheric CO$_2$ and wind')
quiver(CO2.std('time'), u10.std('time'), v10.std('time'),
       quiverreference_size=2, levels=levels, cmap='viridis', robust=True, ax=axes[0, 1], qkx=.9, qky=.97)
axes[0, 1].set_title(
    'Inter-annual variability surface atmospheric CO$_2$ and wind')
for i, index in enumerate(['positive', 'negative']):
    quiver(comp_CO2.sel(index=index), comp_u10.sel(index=index), comp_u10.sel(
        index=index), ax=axes[1, i], vmin=-1, vmax=1, cmap='RdBu_r', levels=levels, qkx=.47, qky=.47)
axes[1, 0].set_title('El Nino composite')
axes[1, 1].set_title('La Nina composite')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'Figure10_CO2_wind_enso_composites')


for v in ['co2_flux', 'temp2', 'precip', 'co2_flux_cumsum']:
    print(v)
    if v is 'co2_flux_cumsum':
        vdata = xr.open_dataset(_get_path(
            'co2_flux', prefix='control'))['co2_flux'].squeeze()
        vdata = (vdata - vdata.mean('time')).cumsum('time')
        vdata['time'] = np.arange(3000, vdata.time.size + 3000)
    else:
        vdata = xr.open_dataset(_get_path(
            v, prefix='control'))[v].squeeze()

    if 'co2_fl' in v:
        vdata = vdata * 3600 * 24 * 365

    vdata.name = v + ' [' + units[v] + ']'
    ano = (vdata - vdata.mean('time')).compute()

    comp = composite_analysis(
        ano, enso, plot=False, ttest=True, robust=True, yincrease=True, cmap='coolwarm', threshold=threshold)

    levels = 21
    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                             'projection': ccrs.PlateCarree()}, figsize=(15, 7))
    my_plot(vdata.mean('time'), cmap='RdBu_r',
            ax=axes[0, 0], robust=True)
    axes[0, 0].set_title('Mean ' + shortname[v])
    my_plot(vdata.std('time'), levels=levels,
            cmap='viridis', robust=True, ax=axes[0, 1])
    axes[0, 1].set_title('Inter-annual variability ' + shortname[v])
    for i, index in enumerate(['positive', 'negative']):
        my_plot(comp.sel(index=index),
                ax=axes[1, i], robust=True, cmap='RdBu_r', levels=levels)
    axes[1, 0].set_title('El Nino composite')
    axes[1, 1].set_title('La Nina composite')
    plt.tight_layout()
    if savefig:
        plt.savefig(path_paper + 'Figure10_' + v + '_enso_composites')
    plt.show()


# Figs SI diagnostic control
def vwmp_bootstrap(control, l=100, bootstrap=10, sig=99):
    from climpred.bootstrap import varweighted_mean_period_threshold
    d = []
    for _ in range(int(bootstrap / 5)):
        r = np.random.randint(0, control.time.size - l)
        d.append(varweighted_mean_period(control.isel(time=slice(r, r + l))))
    data = xr.concat(d, 'bootstrap')
    data['lon'] = control.lon
    data['lat'] = control.lat
    dpp = data.mean('bootstrap')
    threshold = varweighted_mean_period_threshold(
        control, sig=sig, bootstrap=bootstrap)
    threshold['lon'] = control.lon
    threshold['lat'] = control.lat
    dpp = dpp.where(dpp > threshold)
    dpp.name = 'Period [years]'
    return dpp


def DPP_bootstrap(control, l=100, bootstrap=10, sig=99, **dpp_kwargs):
    """Bootstrap DPP results and mask unsignificant."""
    from climpred.bootstrap import varweighted_mean_period_threshold
    d = []
    # l: length of bootstrap block, block length
    for _ in range(int(bootstrap / 5)):
        r = np.random.randint(0, control.time.size - l)
        d.append(DPP(control.isel(time=slice(r, r + l))))
    data = xr.concat(d, 'bootstrap')
    data['lon'] = control.lon
    data['lat'] = control.lat
    dpp = data.mean('bootstrap')
    threshold = DPP_threshold(
        control, sig=sig, bootstrap=bootstrap, **dpp_kwargs)
    threshold['lon'] = control.lon
    threshold['lat'] = control.lat
    dpp = dpp.where(dpp > threshold)
    dpp.name = 'DPP [years]'
    return dpp


l = 100
bootstrap = 500


def plot_varname_control_diagnostics(v, cmap='viridis'):
    """Plot variability diagnostics from control run.

    - (0,0): mean
    - (0,1): standard deviation
    - (1,0): Diagnostic Potential Predictability DPP m=10
    - (1,1): Variance-weighted mean period
    """
    varname = v
    fig, ax = plt.subplots(ncols=2, nrows=2, subplot_kw={
                           'projection': ccrs.PlateCarree()}, figsize=(14, 7))
    if v is 'co2_flux_cumsum':
        control = xr.open_dataset(_get_path(
            'co2_flux', prefix='control'))['co2_flux'].squeeze()
        control = (control - control.mean('time')).cumsum('time')
    else:
        control = xr.open_dataset(_get_path(
            v, prefix='control'))[v].squeeze()
    if 'co2_f' in v:
        control = control * 3600 * 24 * 365
    control.name = v + ' [' + units[v] + ']'

    my_plot(control.mean('time'), robust=True, ax=ax[0, 0], levels=11)
    ax[0, 0].set_title('mean')
    my_plot(control.std('time'), robust=True, ax=ax[0, 1], levels=11)
    ax[0, 1].set_title('std')

    m = 10
    chunk = True

    dpp = DPP_bootstrap(control, bootstrap=bootstrap, l=l, chunk=False)
    my_plot(dpp, curv=False, ax=ax[1, 0], cmap=cmap, edgecolors='None',
            robust=True, levels=11, vmin=0., vmax=.5, cbar_kwargs={
        'label': 'DPP [ ]'})
    ax[1, 0].set_title(
        'Diagnostic Potential Predictability (m=' + str(m) + ')')

    dpp = vwmp_bootstrap(control, bootstrap=bootstrap, l=l)
    dpp.name = 'Period [years]'
    my_plot(dpp, ax=ax[1, 1], curv=False,
            cmap=cmap, vmin=1, vmax=12, levels=12, cbar_kwargs={
        'label': 'mean Period P$_x$ [years]'})
    ax[1, 1].set_title(
        'Variance-weighted mean Period')
    plt.suptitle(longname[v])
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if savefig:
        plt.savefig(path_paper + 'Figure_SI_control_diagnostics_' + v)


for v in ['co2_flux', 'temp2', 'CO2', 'co2_flux_cumsum']:
    plot_varname_control_diagnostics(v)
