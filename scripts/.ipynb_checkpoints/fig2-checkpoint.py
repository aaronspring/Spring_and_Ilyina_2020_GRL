import glob
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import xarray as xr
from climpred.bootstrap import bootstrap_perfect_model
from climpred.graphics import plot_bootstrapped_skill_over_leadyear
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from basics import (_get_path, labels, longname, path_paper, post_global,
                    shortname, units)

warnings.filterwarnings("ignore")
%matplotlib inline

savefig = False
# savefig = True
save_nc = False
# save_nc = True

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'smaller'
labelsize = 14


comparison = 'm2e'
sig = 99
psig = (100 - sig) / 100
bootstrap = 1000

control = xr.open_dataset(post_global + 'control_diagnosed_co2.nc')
ds = xr.open_dataset(post_global + 'ds_diagnosed_co2.nc')

# rename dims to climpred requirements
if 'ensemble' in ds.dims:
    ds = ds.rename({'ensemble': 'init', 'time': 'lead'})

compute = False
if compute:
    bs_acc = bootstrap_perfect_model(
        ds, control, metric='pearson_r', comparison=comparison, bootstrap=bootstrap, sig=sig)
    if save_nc:
        bs_acc.to_netcdf('../data/results/'+'_'.join(['results', 'global', 'ym', 'metric', 'pearson_r',
                                   'comparison', comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc')
    bs_rmse = bootstrap_perfect_model(
        ds, control, metric='rmse', comparison=comparison, bootstrap=bootstrap, sig=sig)
    if save_nc:
        bs_rmse.to_netcdf('../data/results/'+'_'.join(['results', 'global', 'ym', 'metric', 'rmse',
                                    'comparison', comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc')
else:
    bs_acc = xr.open_dataset('../data/results/'+'_'.join(['results', 'global', 'ym', 'metric', 'pearson_r',
                                       'comparison', comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc')
    bs_rmse = xr.open_dataset('../data/results/'+'_'.join(['results', 'global', 'ym', 'metric', 'rmse',
                                        'comparison', comparison, 'sig', str(sig), 'bootstrap', str(bootstrap)]) + '.nc')


bs = bs_acc[v]


def plot_sig_star(bs, ax, sig=99, plot_legend=False):
    """Plot a marker for significant lead years."""
    psig = (100 - sig) / 100
    p_uninit_over_init = bs.sel(kind='uninit', results='p')
    p_uninit_over_init
    init_skill = bs.sel(kind='init', results='skill')
    beat_uninit = init_skill.where(p_uninit_over_init <= psig)
    try:
        ph = int(init_skill.where(p_uninit_over_init <=
                                  psig).notnull().argmin().values)
    except:
        ph = 0
    ax.scatter(bs.lead.values[:ph + 1], beat_uninit[:ph + 1], marker='*',
               c='k', s=125, zorder=10, label='significant at ' + str(sig) + '% level')
    if plot_legend:
        plt.legend(frameon=False)


def plot_both_skill(bs_acc, bs_rmse, ax=None, unit='unit', not_all=True, sig=99):
    """Plot RMSE and ACC skill on x and y axis."""
    fontsize = 8
    c_uninit = 'indianred'
    c_init = 'steelblue'
    c_pers = 'gray'
    capsize = 4
    p = (100 - sig) / 100  # 0.05
    ci_low = p / 2  # 0.025
    ci_high = 1 - p / 2  # .975
    pers_sig = sig

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    # acc
    init_skill = bs_acc.sel(kind='init', results='skill')
    init_ci = bs_acc.sel(kind='init', results=['low_ci', 'high_ci']).rename(
        {'results': 'quantile'}
    )
    uninit_skill = bs_acc.sel(kind='uninit', results='skill').isel(lead=0)
    uninit_ci = (
        bs_acc.sel(kind='uninit', results=['low_ci', 'high_ci'])
        .rename({'results': 'quantile'})
        .isel(lead=0)
    )
    pers_skill = bs_acc.sel(kind='pers', results='skill')
    pers_ci = bs_acc.sel(kind='pers', results=['low_ci', 'high_ci']).rename(
        {'results': 'quantile'}
    )
    p_uninit_over_init = bs_acc.sel(kind='uninit', results='p')
    p_pers_over_init = bs_acc.sel(kind='pers', results='p')

    # rmse
    init_skill2 = bs_rmse.sel(kind='init', results='skill')
    init_ci2 = bs_rmse.sel(kind='init', results=['low_ci', 'high_ci']).rename(
        {'results': 'quantile'}
    )
    uninit_skill2 = bs_rmse.sel(kind='uninit', results='skill').isel(lead=0)
    uninit_ci2 = (
        bs_rmse.sel(kind='uninit', results=['low_ci', 'high_ci'])
        .rename({'results': 'quantile'})
        .isel(lead=0)
    )
    pers_skill2 = bs_rmse.sel(kind='pers', results='skill')
    pers_ci2 = bs_rmse.sel(kind='pers', results=['low_ci', 'high_ci']).rename(
        {'results': 'quantile'}
    )
    p_uninit_over_init2 = bs_rmse.sel(kind='uninit', results='p')
    p_pers_over_init2 = bs_rmse.sel(kind='pers', results='p')

    # PH
    beat_uninit = init_skill.where(p_uninit_over_init < psig)
    beat_uninit2 = init_skill2.where(p_uninit_over_init2 < psig)
    ph1 = int(init_skill.where(p_uninit_over_init <
                               psig).notnull().argmin().values)
    if ph1 is 0 and beat_uninit.all():
        ph1 = beat_uninit.lead.size
    ph2 = int(init_skill2.where(p_uninit_over_init2 <
                                psig).notnull().argmin().values)
    if ph2 is 0 and beat_uninit2.all():
        ph2 = beat_uninit2.lead.size
    print('PH:: ACC:', ph1, ', RMSE:', ph2)
    ph = min(ph1, ph2)
    ph_max = max(ph1, ph2)
    # make non sig less visible
    ax.errorbar(
        init_skill2[ph_max:],
        init_skill[ph_max:],
        yerr=[
            init_skill[ph_max:] - init_ci.isel(quantile=0)[ph_max:],
            init_ci.isel(quantile=1)[ph_max:] - init_skill[ph_max:]
        ],
        xerr=[
            init_skill2[ph_max:] - init_ci2.isel(quantile=0)[ph_max:],
            init_ci2.isel(quantile=1)[ph_max:] - init_skill2[ph_max:]
        ],
        fmt='--o',
        alpha=.2,
        capsize=capsize,
        zorder=1,
        c=c_uninit,
        label='')
    # init
    ax.errorbar(
        init_skill2[:ph_max],
        init_skill[:ph_max],
        yerr=[
            init_skill[:ph_max] - init_ci.isel(quantile=0)[:ph_max],
            init_ci.isel(quantile=1)[:ph_max] - init_skill[:ph_max]
        ],
        xerr=[
            init_skill2[:ph_max] - init_ci2.isel(quantile=0)[:ph_max],
            init_ci2.isel(quantile=1)[:ph_max] - init_skill2[:ph_max]
        ],
        fmt='--o',
        capsize=capsize,
        c=c_uninit,
        label='initialized at ' + str(sig) + '% confidence interval')
    # uninit
    ax.errorbar(
        uninit_skill2,
        uninit_skill,
        xerr=[[uninit_skill2 - uninit_ci2.isel(quantile=0)],
              [uninit_ci2.isel(quantile=1) - uninit_skill2]],
        yerr=[[uninit_skill - uninit_ci.isel(quantile=0)],
              [uninit_ci.isel(quantile=1) - uninit_skill]],
        fmt='--o',
        capsize=capsize,
        c=c_init,
        zorder=4,
        label='uninitialized at ' + str(sig) + '% confidence interval')

    ax.scatter(init_skill2[:ph1], init_skill[:ph1], marker='*',
               c='gray', s=125, alpha=.5, zorder=10, label='')
    ax.scatter(init_skill2[:ph2], init_skill[:ph2], marker='*',
               c='gray', s=125, alpha=.5, zorder=10, label='')
    ax.scatter(init_skill2[:ph], init_skill[:ph], marker='*',
               c='k', s=125, zorder=10, label='significant at ' + str(sig) + '% level')
    # format
    ax.axvline(x=uninit_skill2, c='steelblue', ls=':')
    ax.axhline(y=uninit_skill, c='steelblue', ls=':')
    if not_all:
        ax.legend(frameon=False, handletextpad=0.05)
        ax.set_ylabel('ACC')
        ax.set_ylim(top=1.)
        ax.set_xlabel('RMSE [' + unit + ']')


vlist = ['co2_flx_ocean', 'co2_flx_land', 'CO2',
         'diag_CO2_ocean', 'diag_CO2_land', 'diag_CO2']


def plot_fig2(bs_acc, bs_rmse, label_offset=0, sig=95):
    """Plot RMSE and ACC skill for 6 variables. Spring and Ilyina, 2019 Figure 2."""
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(
        15, 6), sharey='row', sharex=False)
    for i, ax in enumerate(axes.flatten()[:6]):
        v = vlist[i]
        plot_both_skill(bs_acc[v], bs_rmse[v],
                        ax=ax, unit=units[v], not_all=False, sig=sig)
        ax.set_title(shortname[v])
        ax.set_xlabel('RMSE [' + units[v] + ']')
        # # TODO: fix index label
        ax.add_artist(AnchoredText('(' +
                                   labels[i + label_offset] + ')', prop=dict(size=labelsize), frameon=False, loc=2, pad=.05, borderpad=.1))
    axes[0, 0].set_ylabel('ACC [ ]')
    axes[1, 0].set_ylabel('ACC [ ]')
    axes[0, 2].set_xlim([0, .9])
    axes[1, 2].set_xlim([0, .9])
    axes[0, 0].set_ylim([-.6, 1.1])
    axes[1, 0].set_ylim([-.6, 1.1])
    plt.tight_layout(h_pad=.1)


# plot and save
plot_fig2(bs_acc, bs_rmse)
if savefig:
    plt.savefig(path_paper + 'Figure2_2_co2_flux_skill', bbox_inches='tight')
