import glob
import warnings
from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from climpred.bootstrap import bootstrap_perfect_model

import geopandas
from basics import (_get_path, comply_climpred, data_path, labels, longname,
                    path_paper, post_global, shortname, units)
from PMMPIESM.plot import _plot_co2_stations, my_plot

warnings.filterwarnings("ignore")
%matplotlib inline

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'smaller'
labelsize = 14

save_nc = False

v = 'co2_flux'
control = xr.open_dataset(_get_path(v, prefix='control'))
ds = xr.open_dataset(_get_path(v, prefix='ds'))


ds, control = comply_climpred(ds, control)

comparison = 'm2e'
psig = (100 - sig) / 100
bootstrap = 1000

metric = 'pearson_r'
sig = 95
v = 'co2_flux_cumsum'
metric = 'rmse'
for metric in ['pearson_r', 'rmse']:
    for v in ['co2_flux_cumsum', 'co2_flux', 'CO2']:
        varstring = copy(v)
        if v.endswith('_cumsum'):
            v = v[:-7]
        print(v, varstring)
        ds = xr.open_dataset(data_path + 'ds_' + v + '_ym.nc').load()
        control = xr.open_dataset(data_path + 'control_' + v + '_ym.nc').load()
        ds, control = comply_climpred(ds, control)
        if varstring is 'co2_flux_cumsum':
            print('cumsum')
            # remove mean and aggregate
            a = (ds - control.mean('time')).cumsum('lead')
            b = (control - control.mean('time')).cumsum('time')
            a['lead'] = ds.lead
            b['time'] = control.time
            ds = a
            control = b
        else:  # faster
            ds = ds.isel(lead=slice(None, 10))

        bs = bootstrap_perfect_model(
            ds, control, metric=metric, comparison=comparison, bootstrap=bootstrap, sig=sig)
        if save_nc:
            bs.to_netcdf('_'.join(['results', varstring, 'ym', 'metric', metric, 'comparison',
                                   comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc')


# plots
bootstrap = 1000
comparison = 'm2e'
psig = .05

# prototyping
metric = 'rmse'
v = 'co2_flux'
varstring = copy(v)
if varstring.endswith('cumsum'):
    v = v[:-7]
fname = '_'.join(['results', varstring, 'ym', 'metric', metric, 'comparison',
                  comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc'
ds = xr.open_dataset(fname)[v]
skill = ds.sel(results='skill', kind='init').where(
    ds.sel(results='p', kind='uninit') <= psig)
# skill.plot(col='lead', col_wrap=5)


def ph_from_skill(skill):
    """Get predictability horizon (ph) from skill at last non-nan lead while setting all-nan leads to zero PH. Nans are masked if p below psig."""
    ph = skill.argmin('lead', skipna=False)
    ph = ph.where(ph != 0, np.nan)
    ph_not_reached = (skill.notnull()).all('lead')
    ph = ph.where(~ph_not_reached, other=skill['lead'].max())
    return ph

# ph = ph_from_skill(skill)
# ph.plot()


def plot_ph(ph, vmax=12, vmin=0, cmapstr='viridis', ax=None, curv=False, plot_cb_phname=True, cbar_shrink=1):
    """Plot predictability horizon with stride 1 colormap."""
    ph.name = 'Predictability Horizon [years]'
    ph_levels = [_ for _ in range(vmin, vmax + 1)]
    ncolors = len(ph_levels)
    cmap = _cmap_discretize(cmapstr, ncolors)
    # cmap.set_under('white')
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_array([])
    l_min = max(vmin, np.min(ph_levels))
    l_max = min(np.max(ph_levels), vmax)
    ticks = np.arange(l_min, l_max + 1)
    mappable.set_clim(l_min - 0.5, l_max + 0.5)
    if ax is None:
        p = my_plot(ph.where(ph > 0), levels=ph_levels,
                    add_colorbar=False, cmap=cmapstr, curv=curv)
    else:
        p = my_plot(ph.where(ph > 0), levels=ph_levels,
                    add_colorbar=False, ax=ax, cmap=cmapstr, curv=curv)
    cb = plt.colorbar(mappable, ax=ax, shrink=cbar_shrink)
    if vmax == 20:
        ticks = [0, 5, 10, 15, 20]
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticks)
    cb.ax.set_ylabel('[years]', fontsize=15)
    return p


def plot_fig_lead_ph(skill2, ph, metric, cmap='viridis_r', max_ph=7, varstring=None, robust_level=None, **kwargs):
    """Plot first five leads and predictability horizon as panel plot.

    See Spring and Ilyina, 2019 Figs. 3 & 4."""
    if varstring is None:
        varstring = skill2.name
    projection = ccrs.PlateCarree()
    # faking robust for show contours for land and ocean co2_flux
    # which are at different levels of magnitude
    if robust_level is None:
        if v is 'co2_flux' and metric is 'rmse':
            robust_level_lower = 0.000001
            robust_level_upper = .925
            print('set robust', robust_level_lower, robust_level_upper)
        else:
            robust_level_lower = 0.01
            robust_level_upper = 1 - robust_level_lower
    vmin = skill2.quantile(robust_level_lower).values
    vmax = skill2.quantile(robust_level_upper).values

    if ph.quantile(.98) > 15:
        max_ph = 20
    fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True, figsize=(
        20, 7), subplot_kw={'projection': projection})
    for i, axes in enumerate(ax.flatten()):
        if i == 5:
            pass
        else:
            p = my_plot(skill2.isel(
                lead=i), ax=axes, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        axes.add_artist(AnchoredText('(' + labels[i] + ')', prop=dict(
            size=15), frameon=False, loc=2, pad=.05))
    c = .93
    c_map_ax = fig.add_axes([c - .01, 0.22, 0.015, 0.6])

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap2 = copy(plt.get_cmap(cmap))
    # set unpredictable, masked part of colorbar to white
    if metric is 'pearson_r':
        cmap2.set_under('white', 1.)
    elif metric is 'rmse':
        cmap2.set_over('white', 1.)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    metricname = metric_dict[metric]
    unit = units[varstring]
    if metric is 'pearson_r':
        unit = ' '
    cb = fig.colorbar(sm, cax=c_map_ax, orientation='vertical', extend='both',
                      label=' '.join([shortname[varstring], metricname, '[' + unit + ']']))
    del cmap2
    del sm
    # annotations
    pred_helper_labels = ['unpredictable', 'predictable']
    if metric is 'pearson_r':
        pred_helper_labels = pred_helper_labels[::-1]
    fig.text(c + 0.01, .87, pred_helper_labels[0],
             fontsize=15, horizontalalignment='center')  # top
    fig.text(c, .18, pred_helper_labels[1], fontsize=15,
             horizontalalignment='center')  # bottom

    # plot ph
    axph = ax[1, 2]
    if not cmap.endswith('_r') and isinstance(cmap, str):
        cmap = cmap + '_r'
    assert isinstance(cmap, str)
    plot_ph(ph, ax=axph, plot_cb_phname=False,
            cbar_shrink=.8, vmax=max_ph, cmapstr='pink_r')
    if v == 'CO2':  # plot CO2 station location only when CO2 is varname
        _plot_co2_stations(ph=ph, print_results=True,
                           print_station_names=False, ax=axph)
    axph.set_title('Predictability Horizon')
    # set abc
    axph.add_artist(AnchoredText('(' + labels[-1] + ')', prop=dict(
        size=15), frameon=False, loc=2, pad=.05))

    plt.subplots_adjust(wspace=0.04, hspace=0.15)
    del cmap


plot_fig_lead_ph(skill, ph, metric, cmap='cmo.haline',
                 levels=None, varstring=varstring)

#plot_fig_lead_ph(skill, ph, metric, cmap='cmo.haline', levels=None, varstring=varstring)

# Plotting loop
conv = 24 * 3600 * 365
psig = .05
max_ph = 6
for metric in ['pearson_r', 'rmse']:
    for v in ['co2_flux', 'CO2', 'co2_flux_cumsum']:
        cmap = 'cmo.haline_r'
        varstring = copy(v)
        if varstring.endswith('cumsum'):
            v = v[:-7]
        fname = '_'.join(['results', varstring, 'ym', 'metric', metric, 'comparison',
                          comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc'
        ds = xr.open_dataset(fname)[v]
        skill = ds.sel(results='skill', kind='init').where(
            ds.sel(results='p', kind='uninit') <= psig)

        if metric == 'rmse' and v == 'co2_flux':
            skill = skill * conv
            print('conv')

        ph = ph_from_skill(skill)
        skill2 = skill.isel(lead=slice(None, 5))
        if metric == 'rmse' and isinstance(cmap, str):
            if not cmap.endswith('_r'):
                cmap = cmap + '_r'
            else:
                if isinstance(cmap, str):
                    cmap = cmap[:-2]
        metricname = metric_dict[metric]
        print(v, varstring, metric, metricname, cmap)
        plot_fig_lead_ph(skill2, ph, metric, cmap,
                         max_ph=max_ph, varstring=varstring)
        if savefig:
            plt.savefig(path_paper + 'Figure3_' + varstring +
                        '_' + metricname, bbox_inches='tight')
        plt.show()

# Note filenames contain sig and bootstrap information. However, the siglevel there is only important for low_ci and high_ci, but for my calculatation I use only skill and mask p where lower than psig.

# extract Predictability Horizon table for latex
metric = 'rmse'
v = 'CO2'
sig = 99
bootstrap = 1000
comparison = 'm2e'
varstring = copy(v)

fname = '_'.join(['results', varstring, 'ym', 'metric', metric, 'comparison',
                  comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc'
psig = .05
ds = xr.open_dataset(fname)[v]
skill = ds.sel(results='skill', kind='init').where(
    ds.sel(results='p', kind='uninit') <= psig)


def ph_from_skill(skill):
    ph = skill.argmin('lead', skipna=False)
    ph = ph.where(ph != 0, np.nan)
    ph_not_reached = (skill.notnull()).all('lead')
    ph = ph.where(~ph_not_reached, other=skill['lead'].max())
    return ph


# extract Predictability Horizon table
ph = ph_from_skill(skill)

plot_ph(ph)
df_ph_rmse = _plot_co2_stations(ph)

metric = 'pearson_r'
fname = '_'.join(['results', varstring, 'ym', 'metric', metric, 'comparison',
                  comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc'
psig = .05
ds = xr.open_dataset(fname)[v]
skill = ds.sel(results='skill', kind='init').where(
    ds.sel(results='p', kind='uninit') <= psig)
ph = ph_from_skill(skill)
plot_ph(ph)
df_ph_acc = _plot_co2_stations(ph)
df_ph_acc

df_ph = pd.merge(df_ph_rmse, df_ph_acc, how='outer').T

df_ph.columns = ['Lon', 'Lat', 'PH RMSE', 'PH ACC']

df_ph = df_ph.astype('int')

if save_nc:
    with open('ph_co2_stations.txt', 'w') as tf:
        tf.write(df_ph.to_latex())
