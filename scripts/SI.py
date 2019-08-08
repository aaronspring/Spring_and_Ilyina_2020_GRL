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
from scripts.basics import (_get_path, comply_climpred, labels, longname,
                            metric_dict, path_paper, post_global, post_ML,
                            shortname, units)
from xskillscore import pearson_r

import cmocean
from esm_analysis.composite import composite_analysis
from esm_analysis.stats import rm_trend
from PMMPIESM.plot import my_plot

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

# difference in predictability horizon


def step_lower(x, a):
    return 1 * (x < a)


def step_higher(x, a):
    return 1 * (x >= a)


def func(t, b, ph, c):
    f = step_lower(t, ph) * ((c - b) / (ph) * (t) + b) + step_higher(t, ph) * c
    return f


def fit_ph_int(s, plot=False):
    """
    Calculate quick & dirty predictability horizon (PH) by optimizing squared differences.

    input: pd.series
    output: b, ph, c
    """
    import scipy.optimize as optimization
    t_fit = np.arange(0, s.size)
    sigmav = s.std()
    sigma = np.linspace(sigmav / 2, sigmav, s.size)
    mean = s
    d = []
    e = []
    w = []
    b = s.iloc[0]
    x0 = np.array([0., s[5:].mean()])
    for ph in t_fit:

        def func(t, b, c):
            f = step_lower(t, ph) * ((c - b) / (ph - 1)
                                     * (t - 1) + b) + step_higher(t, ph) * c
            return f

        fit = optimization.curve_fit(func, t_fit, mean, x0, sigma)
        b = fit[0][0]
        c = fit[0][1]
        a = func(t_fit, b, c)
        diff2 = np.sum((mean - a)**2)
        if ph < 15:
            d.append(diff2)
            e.append(b)
            w.append(c)

    def func(t, b, ph, c):
        f = step_lower(t, ph) * ((c - b) / (ph - 1)
                                 * (t - 1) + b) + step_higher(t, ph) * c
        return f

    ph = np.nanargmin(d)
    c = w[ph]
    b = e[ph]
    return b, ph + 1, c


def predictability_horizon(skill):
    """Get predictability horizon (ph) from skill at last non-nan lead while setting all-nan leads to zero PH. Nans are masked if p below psig."""
    ph = skill.argmin('lead', skipna=False)
    ph = ph.where(ph != 0, np.nan)
    ph_not_reached = (skill.notnull()).all('lead')
    ph = ph.where(~ph_not_reached, other=skill['lead'].max())
    return ph


def Sef2018_Fig1_Different_PH_Definitions(ds, control, unit='PgC/yr', sig=95, bootstrap=250, save=True):
    # from esmtools.prediction import predictability_horizon
    from PMMPIESM.plot import _set_integer_xaxis
    rsig = (100 - sig)/100
    _control = control
    _ds = ds
    ss = compute_perfect_model(
        _ds, _control, metric='rmse', comparison='m2e')
    ss['lead'] = np.arange(1, ss.lead.size + 1)
    # ss.name = 'every'
    ss_boot = bootstrap_perfect_model(_ds, _control, metric='rmse',
                                      comparison='m2e', sig=sig, bootstrap=bootstrap)
    ss_p = ss_boot.sel(kind='uninit', results='p')
    ss_ci_high = ss_boot.sel(kind='uninit', results='low_ci')

    ph_Spring_2019 = predictability_horizon(
        ss.where(ss_p < rsig)).values

    b_m2e, ph_Sef_2018, c_m2e = fit_ph_int(ss.to_series())
    print('ph_Sef_2018', ph_Sef_2018)
    print('ph_Spring_2019', int(ph_Spring_2019))

    fig, ax = plt.subplots(figsize=(10, 4))
    std = _control.std('time').values

    every_color = 'mediumorchid'
    ss.name = 'skill'
    ss.to_dataframe().plot(ax=ax, label='skill', color='k', marker='o')

    t_fit = np.arange(0, _ds.lead.size)
    ax.plot(t_fit[1:], func(t_fit, b_m2e, ph_Sef_2018, c_m2e)[1:],
            linewidth=3, color=every_color, label='Sef 2018 breakpoint fit')
    ax.axvline(x=ph_Sef_2018, linestyle='-.',
               color=every_color, label='PH Sef 2018')
    ax.axhline(y=std, ls='--', c='k', alpha=.3, label='std control')
    ax.axhline(y=ss_ci_high.mean('lead'), ls=':',
               c='royalblue', label='Bootstrapped high CI')

    ax.axvline(x=ph_Spring_2019, ls='-.', c='royalblue',
               label='PH Spring 2019')
    ax.set_xlabel('Lead Time [time]')
    ax.set_ylabel('RMSE [' + unit + ']')
    ax.set_ylim([0, ss.max() * 1.1])
    ax.set_xlim([0, 10])
    _set_integer_xaxis(ax)
    ax.legend(frameon=False, ncol=2)
    ax.set_xticks(range(1, 11))
    ax.set_title(
        ' Global oceanic CO$_2$ flux: Differences in definitions of Predictability Horizon')
    if save:
        plt.tight_layout()
        plt.savefig('Differences_PH_definition')


ds = xr.open_dataset('data/results/ds_diagnosed_co2.nc')['co2_flx_ocean']
control = xr.open_dataset(
    'data/results/control_diagnosed_co2.nc')['co2_flx_ocean']
ds, control = comply_climpred(ds, control)


Sef2018_Fig1_Different_PH_Definitions(ds, control)


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


for v in ['co2_flux']  # , 'temp2', 'precip', 'co2_flux_cumsum']:
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
