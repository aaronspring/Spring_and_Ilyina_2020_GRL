# Data

Only globally-averaged output is provided. Please approach me for 3D files personally.

## Primary data from model output used for analysis

-   ensemble simulations global annual timeseries: `ds_{varname}_{timmean}.nc`
-   control simulations global annual timeseries: `control_{varname}_{timmean}.nc`
-   control serves as member 0 in `ds`-file
-   cmorized varnames:
    -   `CO2` converted to `ppm`
    -   `co2_flx_land`: converted to `PgC/yr`
    -   `co2_flx_ocean`: converted to `PgC/yr`

## (intermediate) results

-   ensemble simulations global annual timeseries for Figure 1a-c: `ds_diagnosed_co2.nc`
-   control global annual timeseries for Figure 1a-c: `control_diagnosed_co2.nc`
-   Predictability Horizon for CO2 measurement stations from `fig2_4.py`: `ph_co2_stations.txt`
-   Bootstrapped global annual predictive skill results for Figure 2: `results_{name..region}_metric_{metric_name}_comparison_{comparison_name}_sig_95_bootstrap_5000.nc`
