from PMMPIESM.setup import PM_path

labels = 'abcdefghi'

units = dict()
units['co2_flux'] = 'PgC/yr'
units['co2_flx_ocean'] = 'PgC/yr'
units['co2_flx_land'] = 'PgC/yr'
units['CO2'] = 'ppm'
units['diag_CO2'] = 'ppm'
units['diag_CO2_ocean'] = 'ppm'
units['diag_CO2_land'] = 'ppm'
units['co2_flux'] = 'gC m$^{-2}$ yr$^{-1}$'
units['co2_flux_cumsum'] = 'gC m$^{-2}$'
units['temp2'] = '$^\circ$C'
units['nino34'] = '$^\circ$C'
units['precip'] = 'kg m$^{-2}$ s$^{-1}$'

longname = dict()
longname['CO2'] = 'surface atmospheric CO$_2$ concentration'
longname['co2_flux'] = 'surface CO$_2$ flux'
longname['co2_flx_ocean'] = 'air-sea CO$_2$ flux'
longname['co2_flx_land'] = 'air-land CO$_2$ flux'
longname['diag_CO2'] = 'diagnosed atm. CO$_2$'
longname['diag_CO2_ocean'] = 'diagnosed atm. CO$_2$ due to accum. air-sea CO$_2$ flux'
longname['diag_CO2_land'] = 'diagnosed atm. CO$_2$ due to accum. air-land CO$_2$ flux'
longname['co2_flux_cumsum'] = 'accumulated surface CO$_2$ flux'
longname['temp2'] = '2m Temperature'
longname['nino34'] = 'Nino 3.4 index'
longname['precip'] = 'Precipitation'

shortname = dict()
shortname['CO2'] = 'prog. CO$_{2,atm}$'
shortname['CO2@3400m'] = 'prog. CO$_{2,atm} @ 3400m$'
shortname['co2_flux'] = 'CO$_2$ flux'
shortname['co2_flux_cumsum'] = 'accum. CO$_2$ flux'
shortname['co2_flx_ocean'] = 'air-sea CO$_2$ flux'
shortname['co2_flx_land'] = 'air-land CO$_2$ flux'
shortname['diag_CO2'] = 'diag. CO$_{2,atm}$'
shortname['diag_CO2_ocean'] = 'diag. CO$_{2,atm,ocean}$'
shortname['diag_CO2_land'] = 'diag. CO$_{2,atm,land}$'
shortname['nino34'] = 'Nino 3.4'
shortname['temp2'] = '2m Temp'
shortname['precip'] = 'Precip.'


metric_dict = dict()
metric_dict['pearson_r'] = 'ACC'
metric_dict['rmse'] = 'RMSE'

path_paper = '/Users/aaron.spring/PhD_Thesis/My_Paper/atmco2_predictability/'
post_global = '/Users/aaron.spring/mistral_m300524/experiments/postprocessed/global/'
post_ML = '/Users/aaron.spring/mistral_m300524/experiments/postprocessed/Mauna_Loa/'
data_path = '/Users/aaron.spring/Desktop/mpi-esm/PM/'


def _get_path(varname=None, exp='PM', prefix='ds', ta='ym', **kwargs):
    """Return string of postprocessed file."""
    if exp is 'PM':
        path = PM_path + 'postprocessed/'
    elif exp is 'GE':
        path = my_GE_path + 'postprocessed/'

    if varname is None:
        raise ValueError('specify varname')

    suffix = ''
    if prefix not in ['ds', 'control']:
        for key, value in kwargs.items():
            if prefix in ['skill']:
                if str(key) in ['sig', 'bootstrap']:
                    continue
            if isinstance(value, str):
                suffix += "_" + key + "_" + str(value)
            else:
                suffix += "_" + key + "_" + str(value)

    filename = prefix + '_' + varname + '_' + ta + suffix + '.nc'
    full_path = path + filename
    return full_path


def comply_climpred(ds, control):
    """Convert dims to use in climpred to calculate skill."""
    if 'ensemble' in ds.dims:
        ds = ds.rename({'ensemble': 'init'})
    if 'year' in ds.dims:
        ds = ds.rename({'year': 'lead'})
    if 'time' in ds.dims:
        ds = ds.rename({'time': 'lead'})
    if 'year' in control.dims:
        control = control.rename({'year': 'time'})
    if 'lev' in ds.coords:
        del ds['lev']
    if 'lev' in control.coords:
        del control['lev']
    return ds, control
