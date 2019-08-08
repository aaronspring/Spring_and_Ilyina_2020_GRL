import warnings
from collections import OrderedDict
from copy import copy

import cartopy.crs as ccrs
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
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scripts.basics import (_get_path, labels, longname, metric_dict,
                            path_paper, post_global, post_ML, shortname, units,
                            yearmonmean)
from xskillscore import pearson_r

import cmocean
from esmtools.composite import composite_analysis
from esmtools.stats import linregress, rm_trend
from PMMPIESM.plot import (_cmap_discretize, _get_PH_station,
                           _plot_co2_stations, my_facetgrid, my_plot, plot_ph,
                           plot_timeseries, truncate_colormap)
from PMMPIESM.predictability import bootstrap_predictability_horizon
from PMMPIESM.setup import load_reg_area

warnings.filterwarnings("ignore")
%matplotlib inline
# mpl.rcParams.keys()
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
#mpl.rcParams['colorbar.titlesize'] = 'medium'

r_ppmw2ppmv = 28.8 / 44.0095
CO2_to_C = 44.0095 / 12.0111

savefig = False
# savefig = True
save_nc = False
# save_nc = True

ds = xr.open_dataset(_get_path(varname='CO2'))
area = load_reg_area()
weights = area / area.sum(['lat', 'lon'])
ds_atmco2 = (ds * weights).sum(['lat', 'lon'])

control = xr.open_dataset(_get_path(varname='CO2', prefix='control'))
control_atmco2 = (
    control * weights).sum(['lat', 'lon'])


def diagnose_atmco2_PM_control(control):
    """Diagnose atm. CO2 variability from internal variability of accumulated carbon sinks.

    Formula: XCO_{2,\text{atm,diag,total}}(t) = base + \sum_{t'}^t CO_2 flux'_{land+ocean}(t') \cdot \frac{ppm}{2.124 PgC}

    Calculate based on monthly output. See Spring and Ilyina, 2019 methods and supplementary information."""
    base = (control['CO2']).mean(
        'time')
    control['diag_CO2'] = base + (control['co2_flux'] -
                                  control['co2_flux'].mean('time')).cumsum('time') / 2.12
    control['diag_CO2_land'] = base + \
        (control['co2_flx_land'] -
         control['co2_flx_land'].mean('time')).cumsum('time') / 2.12
    control['diag_CO2_ocean'] = base + \
        (control['co2_flx_ocean'] -
         control['co2_flx_ocean'].mean('time')).cumsum('time') / 2.12
    return control


def diagnose_atmco2_PM_ds(ds, control):
    """Diagnose ensemble atm. CO2 variability from internal variability of ensemble accumulated carbon sinks."""
    base = 0
    ds['diag_CO2'] = base + \
        (ds['co2_flux'] - control['co2_flux'].mean('time')).cumsum('time') / 2.12
    ds['diag_CO2_land'] = base + \
        (ds['co2_flx_land'] - control['co2_flx_land'].mean('time')
         ).cumsum('time') / 2.12
    ds['diag_CO2_ocean'] = base + \
        (ds['co2_flx_ocean'] - control['co2_flx_ocean'].mean('time')
         ).cumsum('time') / 2.12
    return ds


def rm_bias(ds, control):
    """Remove bias by removing first lead year mean and adding that years level from control to adjust ds to control levels."""
    for v in ['diag_CO2', 'diag_CO2_land', 'diag_CO2_ocean']:
        ds[v] = ds[v] - yearmonmean(ds[v]).isel(time=0).mean('member') + yearmonmean(
            control[v]).sel(time=ds.ensemble.values).rename({'time': 'ensemble'})
    return ds


recalc_diagnosed = False
if recalc_diagnosed:
    x = []
    for v in ['co2_flux', 'co2_flx_land', 'co2_flx_ocean', 'CO2']:
        x.append(xr.open_dataset(post_global + 'control_' + v + '_mm.nc')[v])
    control_global = xr.merge(x).drop('lev')
    control_global = diagnose_atmco2_PM_control(control_global)

    x = []
    for v in ['co2_flux', 'co2_flx_land', 'co2_flx_ocean', 'CO2']:
        x.append(xr.open_dataset(post_global + 'ds_' + v + '_mm.nc')[v])
    ds_global = xr.merge(x).drop('lev')
    # diagnose
    ds_global = diagnose_atmco2_PM_ds(ds_global, control_global)
    # remove bias
    ds_global = rm_bias(ds_global, control_global)
    # annual averaging
    ds_global = yearmonmean(ds_global)
    co2_flux_varnames = ['co2_flux', 'co2_flx_land', 'co2_flx_ocean']
    # convert to yearsum
    for v in co2_flux_varnames:
        ds_global[v] = ds_global[v] * 12
    ds_global['time'] = np.arange(1, 1 + ds_global.time.size)
    control_ = yearmonmean(control_global)
    co2_flux_varnames = ['co2_flux', 'co2_flx_land', 'co2_flx_ocean']
    # make yearsum for fluxes
    for v in co2_flux_varnames:
        control_global[v] = control_global[v] * 12

    ds_global.to_netcdf(post_global + 'ds_diagnosed_co2.nc')
    control_global.to_netcdf(post_global + 'control_diagnosed_co2.nc')
else:
    ds_global = xr.open_dataset(post_global + 'ds_diagnosed_co2.nc')
    control_global = xr.open_dataset(post_global + 'control_diagnosed_co2.nc')


# Figure 1a-c
def plot_timeseries(ds,
                    control,
                    ignore=True,
                    ax=False,
                    ens_color='mediumseagreen',
                    ensmean_color='seagreen',
                    mean_line=False):
    """Plot an ensemble timeseries. Ignore a list of ensembles.

    Args
        ds, control : xr.Dataset
        varname : str
    """
    ignore_ens = [3023, 3124, 3139, 3178, 3237]
    if not ax:
        fig, ax = plt.subplots(figsize=(20, 5))

    # plot control
    control = control.to_series()
    ax.plot(control, color='black', alpha=1, label='control')
    if mean_line:
        ax.axhline(control.mean(), color='blue', alpha=.1)

    # plot ens, ensmean, vertical lines
    for ens in ds.ensemble.values:
        if ignore and ens in ignore_ens:
            continue
        ax.axvline(
            x=ens - 1,
            color='black',
            alpha=.2,
            linestyle='--',
            label='ensemble initialization')
        df = ds.sel(
            ensemble=ens).to_dataframe('varname').unstack()['varname'].T
        df[0] = control.loc[ens - 1]
        df = df.T.sort_index(axis=0)
        df.index = np.arange(ens - 1, ens - 1 + df.index.size)
        ax.plot(df, color=ens_color, linewidth=0.5, label='ensemble members')
        ax.plot(
            df.mean(axis=1),
            color=ensmean_color,
            linewidth=2,
            alpha=1,
            label='ensemble mean')

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=4,
              prop={'size': 12}, loc='lower center')


# Fig. 1ab
fig, ax2 = plt.subplots(nrows=2,
                        ncols=1, figsize=(11, 5), sharex=True)
i = 0
# set prog to diag level, use free parameter
prog = control_global['CO2']
diag = control_global['diag_CO2']
diff = -prog.mean('time')+diag.mean('time')
prog = prog+diff
prog.plot(c='black', label='prognostic CO$_2$', ax=ax2[i])
diag.plot(c='goldenrod', label='diagnostic CO$_2$', ax=ax2[i])
control_global['diag_CO2_land'].plot(
    c='green', label='diag. due to land', ax=ax2[i])
control_global['diag_CO2_ocean'].plot(
    c='royalblue', label='diag. due to ocean', ax=ax2[i])
ax2[i].set_title('diagnostic atm. CO$_2$ method verification')
ax2[i].set_ylabel('[ppm]')
ax2[i].legend(ncol=4, fontsize=12)
ax2[i].set_yticks([278, 279, 280, 281])
ax2[i].set_ylim([277.3, 281.5])
ax2[i].set_xlabel('')
ax2[i].add_artist(AnchoredText('(' + labels[i] + ')', prop=dict(
    size=12), frameon=False, loc=2, pad=.05))
v = 'CO2'
i = 1
plot_timeseries(ds_global[v]+diff.values, prog, ax=ax2[i])
# ax[i].set_title(v)
ax2[i].set_xlim([3000, 3300])
ax2[i].set_ylabel(' [' + units[v] + ']')
ax2[i].set_title(longname[v])
ax2[i].set_yticks([278, 279, 280, 281])
ax2[i].set_ylim([277.3, 281.5])
ax2[i].add_artist(AnchoredText('(' + labels[i] + ')', prop=dict(
    size=12), frameon=False, loc=2, pad=.05))
ax2[i].set_xlabel('Time [year]')
plt.tight_layout(h_pad=.1)
plt.subplots_adjust(top=0.92)
if savefig:
    plt.savefig(path_paper + 'Figure1_global_timeline_overview')


# Fig 1de


def vwmp_bootstrap(control, l=100, bootstrap=10, sig=99):
    """Bootstrap variance weighted mean period from control. Masked areas are non-significant. Rejected at 1-sig% level."""
    from climpred.bootstrap import varweighted_mean_period_threshold
    d = []
    for _ in range(int(bootstrap / 5)):
        r = np.random.randint(0, control.time.size - l)
        d.append(varweighted_mean_period(control.isel(time=slice(r, r + l))))
    data = xr.concat(d, 'bootstrap')
    data['lon'] = control.lon
    data['lat'] = control.lat
    vwmp = data.mean('bootstrap')
    threshold = varweighted_mean_period_threshold(
        control, sig=sig, bootstrap=bootstrap)
    threshold['lon'] = control.lon
    threshold['lat'] = control.lat
    vwmp = vwmp.where(vwmp > threshold)
    vwmp.name = 'Period [years]'
    return vwmp


params = {'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small',
          }
mpl.rcParams.update(params)
l = ['surface annual CO$_2$ flux',
     'surface annual prognostic atm. CO$_2$ mixing ratio']


bootstrap = 500
v = 'co2_flux'
control = xr.open_dataset(_get_path(v, prefix='control'))[v]
vwmp_co2_flux = vwmp_bootstrap(control, bootstrap=bootstrap)

v = 'CO2'
control = xr.open_dataset(_get_path(v, prefix='control'))[v].drop('lev')
vwmp_CO2 = vwmp_bootstrap(control, bootstrap=bootstrap)

vwmp = xr.concat([vwmp_co2_flux, vwmp_CO2], 'variable')

g = my_facetgrid(vwmp, col='variable', robust=True, cmap='cmo.haline', aspect=2, cbar_kwargs={
                 'label': 'mean Period P$_x$ [years]'}, vmin=2, vmax=12, plot_lon_lat_axis=True)
for i, ax in enumerate(g.axes.flat):
    ax.set_title(l[i], fontsize=14)
    ax.add_artist(AnchoredText('(' + labels[i+2] + ')', prop=dict(
        size=12), frameon=False, loc=2, pad=.05))
if savefig:
    plt.savefig(path_paper + 'Figure1_vwmp')


# Fig SI Mauna Loa 1
ds_ML = xr.open_dataset(post_ML + 'ds_CO2_ym.nc')
control_ML = xr.open_dataset(post_ML + 'control_CO2_ym.nc')
fig, ax = plt.subplots(figsize=(12, 3))
plot_timeseries(ds_ML['CO2'], control_ML['CO2'], ax=ax)
plt.title('annual atmospheric CO$_2$ concentrations at Mauna Loa')
plt.ylabel('atmospheric CO$_2$ [ppm]')
plt.xlabel('Time [year]')
plt.xlim([3000, 3300])
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'Figure1_atmco2_timeline_Mauna_Loa')
