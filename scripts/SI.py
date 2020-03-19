import warnings
from collections import OrderedDict

import climpred
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from climpred.bootstrap import (DPP_threshold, bootstrap_perfect_model,
                                varweighted_mean_period_threshold)
from climpred.prediction import compute_perfect_model, compute_persistence
from climpred.stats import DPP, autocorr, rm_trend, varweighted_mean_period
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from pandas.plotting import autocorrelation_plot
from scripts.basics import (_get_path, comply_climpred, labels, longname,
                            metric_dict, path_paper, post_global, post_ML,
                            shortname, units, yearmonmean)
from xskillscore import pearson_r

import cartopy.crs as ccrs
import cmocean
import PMMPIESM
from esm_analysis.composite import composite_analysis
from esm_analysis.stats import rm_trend
from PMMPIESM.plot import my_plot

warnings.filterwarnings("ignore")
%matplotlib inline

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['savefig.format'] = 'eps'

r_ppmw2ppmv = 28.8 / 44.0095
CO2_to_C = 44.0095 / 12.0111

savefig = False
savefig = True


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


# Load global data
post_global = 'data/results/'
ds_global = xr.open_dataset(post_global + 'ds_diagnosed_co2.nc')
control_global = xr.open_dataset(post_global + 'control_diagnosed_co2.nc')

fig, ax2 = plt.subplots(nrows=2,
                        ncols=1, figsize=(11, 5), sharex=True)
v = 'CO2'
i = 0
plot_timeseries(ds_global[v], control_global[v], ax=ax2[i])
ax2[i].set_xlim([3000, 3300])
ax2[i].set_ylabel(' [' + units[v] + ']')
ax2[i].set_title('global '+longname[v])
ax2[i].set_yticks([278, 279, 280, 281, 282])
ax2[i].set_ylim([277.3, 282.5])
ax2[i].add_artist(AnchoredText('(' + labels[i] + ')', prop=dict(
    size=12), frameon=False, loc=2, pad=.05))
ax2[i].get_legend().remove()

# Load ML data
p = 'data/plain_model_output/Mauna_Loa/'
control = xr.open_dataset(p + 'control_CO2_mm.nc').compute()
ds = xr.open_dataset(p + 'ds_CO2_mm.nc').compute()
del ds['lev']
del control['lev']
ds = yearmonmean(ds)
ds['time'] = np.arange(1, 1+ds.time.size)
control = yearmonmean(control)
control['time'] = np.arange(3000, 3000+control.time.size)


i = 1
plot_timeseries(ds[v], control[v], ax=ax2[i])
ax2[i].set_xlim([3000, 3300])
ax2[i].set_ylabel(' [' + units[v] + ']')
ax2[i].set_title('Mauna Loa atmospheric CO$_2$ mixing ratio')
ax2[i].set_yticks([278, 279, 280, 281, 282])
ax2[i].set_ylim([277.3, 282.5])
ax2[i].add_artist(AnchoredText('(' + labels[i] + ')', prop=dict(
    size=12), frameon=False, loc=2, pad=.05))
ax2[i].set_xlabel('Time [year]')
plt.tight_layout(h_pad=.1)
plt.subplots_adjust(top=0.92)
if savefig:
    plt.savefig(path_paper + 'FigureSI_timeline_prog_CO2')


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


def Sef2018_Fig1_Different_PH_Definitions(ds, control, unit='PgC/yr', sig=95, bootstrap=1000):
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
    if savefig:
        plt.tight_layout()
        plt.savefig('FigureSI_Differences_PH_definition')


ds = xr.open_dataset('data/results/ds_diagnosed_co2.nc')['co2_flx_ocean']
control = xr.open_dataset(
    'data/results/control_diagnosed_co2.nc')['co2_flx_ocean']
ds, control = comply_climpred(ds, control)


Sef2018_Fig1_Different_PH_Definitions(ds, control)


# composite_analysis
mpl.rcParams['savefig.format'] = 'jpg'
mpl.rcParams['savefig.dpi'] = 600
# Load ENSO
control3d = xr.open_dataset(_get_path(
    'tos', prefix='control'))
control3d = control3d - \
    control3d.rolling(time=10, center=True, min_periods=1).mean()

nino34_mask = xr.open_dataset(
    PMMPIESM.setup.file_origin + 'masks/enso_34_mask.nc')['alon'].squeeze()
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
       quiverreference_size=10, levels=levels, robust=True, cmap='viridis', ax=axes[0, 0], qkx=.45, qky=.95)

axes[0, 0].set_title('Mean surface atmospheric CO$_2$ and wind')
quiver(CO2.std('time'), u10.std('time'), v10.std('time'),
       quiverreference_size=2, levels=levels, cmap='viridis', robust=True, ax=axes[0, 1], qkx=.95, qky=.95)
axes[0, 1].set_title(
    'Inter-annual variability surface atmospheric CO$_2$')
for i, index in enumerate(['positive', 'negative']):
    quiver(comp_CO2.sel(index=index), comp_u10.sel(index=index), comp_u10.sel(
        index=index), ax=axes[1, i], vmin=-1, vmax=1, cmap='RdBu_r', levels=levels, qkx=.45, qky=.47)
axes[1, 0].set_title('El Nino composite')
axes[1, 1].set_title('La Nina composite')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_CO2_wind_enso_composites')

for v in ['co2_flux']:  # , 'temp2', 'precip', 'co2_flux_cumsum']:
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
        plt.savefig(path_paper + 'FigureSI_' + v + '_enso_composites')
    plt.show()


# re-emergence enso check
p = '/Users/aaron.spring/PhD_Thesis/PhD_scripts/180724_perfect_model_predictability/'
ds = xr.open_dataset(p+'ds_period_area_ens_m_var.nc').sel(period='ym',
                                                          area='Tropical_Pacific').rename({'year': 'time'})
ds['time'] = np.arange(1, 1+ds.time.size)
control = xr.open_dataset(
    p+'control_period_area_var.nc').sel(period='ym', area='Tropical_Pacific').rename({'year': 'time'})
control['time'] = np.arange(3000, 3000+control.time.size)
v = 'nino34'
plot_timeseries(ds[v], control[v])
plt.xlim([3000, 3300])
plt.ylabel('Nino 3.4 index [ ]')
plt.title('Nino 3.4 timeseries')
plt.ylim([-2, 2])
plt.xlim([3140, 3280])
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_' + v + '_timeseries')


for i in ds.ensemble.values:
    ds['nino34'].isel(time=slice(None, 5)).std('member').sel(
        ensemble=i).rename({'time': 'lead'}).plot(label=i)
plt.legend(loc='upper right', fontsize='x-small')
plt.title('Nino 3.4 member RMSE')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_' + v + '_std')


autocorrelation_plot(control['nino34'])
plt.xlim([0, 10])
plt.title('Nino 3.4')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_' + v + '_ACF')

area = '35S-35N'
ds = xr.open_dataset(p+'ds_period_area_ens_m_var.nc').sel(period='ym',
                                                          area=area).rename({'year': 'time'})
ds['time'] = np.arange(1, 1+ds.time.size)
control = xr.open_dataset(
    p+'control_period_area_var.nc').sel(period='ym', area=area).rename({'year': 'time'})
control['time'] = np.arange(3000, 3000+control.time.size)
v = 'co2_flx_land'
autocorrelation_plot(control[v])
plt.xlim([0, 10])
plt.title(f'Tropical {area} terrestrial CO$_2$ flux')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_' + v + '_ACF')
    plt.show()

for i in ds.ensemble.values:
    ds[v].isel(time=slice(None, 5)).std('member').sel(
        ensemble=i).rename({'time': 'lead'}).plot(label=i)
plt.legend(loc='upper right', fontsize='x-small')
plt.title(f'Tropical {area} terrestrial CO$_2$ flux member RMSE')
plt.ylabel(f'RMSE [{units["co2_flx_land"]}]')
plt.tight_layout()
if savefig:
    plt.savefig(path_paper + 'FigureSI_' + v + '_std')


ds = xr.open_dataset(p+'ds_period_area_ens_m_var.nc').sel(period='ym',
                                                          area='Tropical_Pacific')
control = xr.open_dataset(
    p+'control_period_area_var.nc').sel(period='ym', area='Tropical_Pacific')
ds = ds.rename({'ensemble': 'init', 'year': 'lead'})
ds['lead'] = np.arange(1, 1+ds.lead.size)
# subselect variables
vars = ['nino12', 'nino3', 'nino4', 'nino34', 'co2_flx_land',
        'tsurf', 'precip', 'co2_flx_resp', 'co2_flx_npp', 'atmco2']
ds = ds[vars]
control = control[vars]
units = ['', '', '', '', 'PgC/yr',
         'C', 'm', 'PgC/yr', 'PgC/yr', 'ppm']
len(ds.data_vars)
len(units)
ds.dims
s = compute_perfect_model(ds, ds, metric='rmse', dim='member')


savefig = True
fig, ax = plt.subplots(nrows=6, ncols=2, sharex=True, figsize=(15, 20))
for i, axes in enumerate(ax.flatten()):
    v = list(ds.data_vars)[i]
    s[v].isel(lead=slice(None, 6)).plot.line(
        x='lead', hue='init', ax=axes, add_legend=False)
    s[v].isel(lead=slice(None, 6)).mean('init').plot(
        ax=axes, add_legend=False, color='k', lw=3)
    axes.set_title(v)
    axes.set_ylabel(f'{v} [{units[i]}]')
plt.subplots_adjust(top=.95)
plt.suptitle('Skill for each initialization', fontsize='large')
if savefig:
    plt.savefig(path_paper + 'FigureSI_' + v + '_reemergence')

# subselect anomalous years
dsmm = xr.open_dataset(p+'ds_area_ens_m_var_mm.nc').sel(
    area='Tropical_Pacific').rename({'ensemble': 'init', 'time': 'lead'})
controlmm = xr.open_dataset(
    p+'control_area_ens_m_var_mm.nc').sel(area='Tropical_Pacific')

controlmm = controlmm[vars]
controlmm = climpred.stats.rm_trend(controlmm)

controlmm['time'] = xr.cftime_range(
    start='3000-01', periods=controlmm.time.size, freq='M')


control_init_month = xr.concat(
    [controlmm.sel(time=str(j)).isel(time=0) for j in ds.init.values], 'init')
control_init_month['init'] = dsmm.init

cs = (control_init_month-controlmm.mean('time'))/controlmm.std('time')


boundary = .5
v = 'nino34'

cs[v].plot()
plt.axhline(y=boundary)
plt.axhline(y=-boundary)


anom_pos_nino34_init = cs.where(cs[v] > boundary, drop=True).init
anom_pos_nino34_init.values
anom_neg_nino34_init = cs.where(cs[v] < -boundary, drop=True).init
anom_neg_nino34_init.values
anom_neu_nino34_init = [
    i for i in cs.init.values if i not in anom_pos_nino34_init if i not in anom_neg_nino34_init]
anom_neu_nino34_init
len(anom_pos_nino34_init), len(anom_neu_nino34_init), len(anom_neg_nino34_init)

assert np.sum((len(anom_pos_nino34_init), len(
    anom_neu_nino34_init), len(anom_neg_nino34_init))) == 12
maxlead = 6
sapn = compute_perfect_model(
    ds.sel(init=anom_pos_nino34_init, lead=slice(None, maxlead)), ds, metric='pearson_r')
saneun = compute_perfect_model(
    ds.sel(init=anom_neu_nino34_init, lead=slice(None, maxlead)), ds, metric='pearson_r')
sann = compute_perfect_model(
    ds.sel(init=anom_neg_nino34_init, lead=slice(None, maxlead)), ds, metric='pearson_r')

s = compute_perfect_model(
    ds.isel(lead=slice(None, maxlead)), ds, metric='pearson_r')

sa = xr.concat([s, sapn, saneun, sann], 'IC')
sa['IC'] = ['mean', 'pos', 'neutral', 'neg']

fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(15, 20))
for i, axes in enumerate(ax.flatten()):
    va = list(ds.data_vars)[i]
    sa[va].plot.line(
        x='lead', hue='IC', ax=axes, add_legend=True)
    axes.set_title(va)
    axes.set_ylabel('ACC []')
plt.subplots_adjust(top=.95)
plt.suptitle(
    f'ACC Skill for pos/neutral/neg {v} initial conditions anomaly', fontsize='large')
if savefig:
    plt.savefig(path_paper + 'FigureSI_ACC_reemergence_IC')


v = 'CO2'
# v='co2_flux'
ds3d = xr.open_dataset(_get_path(varname=v)).rename(
    {'ensemble': 'init', 'time': 'lead'})[v].isel(lead=slice(None, 6))
control3d = xr.open_dataset(_get_path(varname=v, prefix='control'))[v]
coarsen = 2
if coarsen:
    ds3d = ds3d.coarsen({'lon': coarsen, 'lat': coarsen}).sum()
    control3d = control3d.coarsen({'lon': coarsen, 'lat': coarsen}).sum()
# ds3d.isel(lead=2,init=2,member=3).plot()
metric = 'pearson_r'
s3d = compute_perfect_model(ds3d, control3d, metric=metric)
# s3d.plot(col='lead')

s3dapn = compute_perfect_model(
    ds3d.sel(init=anom_pos_nino34_init.values), control3d, metric=metric)
s3daneun = compute_perfect_model(
    ds3d.sel(init=anom_neu_nino34_init), control3d, metric=metric)
s3dann = compute_perfect_model(
    ds3d.sel(init=anom_neg_nino34_init.values), control3d, metric=metric)
sa3d = xr.concat([s3d, s3dapn, s3daneun, s3dann], 'IC')
sa3d['IC'] = ['all', 'pos', 'neutral', 'neg']

warnings.simplefilter("ignore")

b = 100
s3d = bootstrap_perfect_model(ds3d, control3d, metric=metric, bootstrap=b)
s3dapn = bootstrap_perfect_model(
    ds3d.sel(init=anom_pos_nino34_init.values), control3d, metric=metric, bootstrap=b)
s3daneun = bootstrap_perfect_model(
    ds3d.sel(init=anom_neu_nino34_init), control3d, metric=metric, bootstrap=b)
s3dann = bootstrap_perfect_model(
    ds3d.sel(init=anom_neg_nino34_init.values), control3d, metric=metric, bootstrap=b)

sa3d.sel(kind='uninit', results='p').plot(col='lead', row='IC', vmax=.2)

sa3d = xr.concat([s3d, s3dapn, s3daneun, s3dann], 'IC')
sa3d['IC'] = ['all', 'pos', 'neutral', 'neg']
skill = sa3d.sel(kind='init', results='skill').where(
    sa3d.sel(results='p', kind='uninit') <= .05)
#sa3d.plot(col='lead', row='IC', robust=True)
#sa3d.name = 'CO$_2$ ACC'

skill = sa3d.isel(lead=slice(None, 5)).where(sa3d > th)

map_proj = ccrs.PlateCarree()
th = .3
p = skill.plot(transform=ccrs.PlateCarree(), col='lead', row='IC', robust=True,
               subplot_kws={"projection": map_proj}, aspect=sa3d["lon"].size / sa3d["lat"].size)
for ax in p.axes.flat:
    ax.coastlines()
    ax.set_aspect("equal")
if savefig:
    plt.savefig(
        f'{path_paper}FigureS12_{v}_{metric.upper()}_3d_reemergence_IC')
plt.show()


dseq = ds3d.sel(lat=slice(5, -5)).mean(['lat', 'lon'])
ceq = control3d.sel(lat=slice(5, -5)).mean(['lat', 'lon'])
compute_perfect_model(eq, control, metric='pearson_r').plot()


mpl.rcParams['savefig.format'] = 'jpg'
# load terrestrial contributions
# v='co2_flux'
ds = xr.Dataset()
control = xr.Dataset()
for v in ['co2_flx_land', 'co2_flx_npp', 'co2_flx_resp', 'co2_flx_herb', 'co2_flx_fire']:
    if v in ds.data_vars:
        continue
    ds3d = xr.open_dataset(_get_path(varname=v)).rename(
        {'ensemble': 'init', 'time': 'lead'})[v].isel(lead=slice(None, 6))
    control3d = xr.open_dataset(_get_path(varname=v, prefix='control'))[v]
    coarsen = 2
    if coarsen:
        ds3d = ds3d.coarsen({'lon': coarsen, 'lat': coarsen}).sum()
        control3d = control3d.coarsen({'lon': coarsen, 'lat': coarsen}).sum()
    ds[v] = ds3d
    control[v] = control3d

ds.data_vars

metric = 'rmse'

s = compute_perfect_model(
    ds, control, metric=metric)

%time bs = bootstrap_perfect_model(ds, control, metric=metric, bootstrap=1000)

bsm = bs.sel(kind='init', results='skill').where(
    bs.sel(results='p', kind='uninit') <= .05)
bsm.to_array().plot(col='lead', row='variable', robust=True)

p = bsm.isel(lead=slice(None, 4)).to_array().plot(transform=ccrs.PlateCarree(), col='lead', row='variable', robust=True,
                                                  subplot_kws={"projection": map_proj}, aspect=bsm["lon"].size / bsm["lat"].size, cbar_kwargs={'label': 'RMSE [gC m$^{-2}$ yr$^{-1}$]'})
for ax in p.axes.flat:
    ax.coastlines()
    ax.set_aspect("equal")
if savefig:
    plt.savefig(f'{path_paper}FigureSI_{metric.upper()}_3d_skill_co2_flx_land')


def ph_from_skill(skill):
    ph = skill.argmin('lead', skipna=False)
    ph = ph.where(ph != 0, np.nan)
    ph_not_reached = (skill.notnull()).all('lead')
    ph = ph.where(~ph_not_reached, other=skill['lead'].max())
    return ph


ph = ph_from_skill(bsm)

ph.to_array().plot(col='variable', levels=[1, 2, 3, 4, 5, 6])


p = ph.to_array().plot(transform=ccrs.PlateCarree(), col='variable', col_wrap=2, robust=True,
                       subplot_kws={"projection": map_proj}, aspect=bsm["lon"].size / bsm["lat"].size, levels=[1, 2, 3, 4], cbar_kwargs={'label': 'Predictability Horizon [Years]'})
for ax in p.axes.flat:
    ax.coastlines()
    ax.set_aspect("equal")
if savefig:
    plt.savefig(f'{path_paper}FigureSI_{metric.upper()}_3d_PH_co2_flx_land')


path_paper
savefig = True
