# Scripts

This repo was setup for scientists interested to reproduce our `Spring and Ilyina, 2019` paper. Please find the used scripts here.

## How to use

-   Setup the environment as described by `requirements.txt`
-   Use the jupyter notebooks (`*.ipynb`) to inspect results interactively; otherwise run `python *.py`
-   First, run `diagnosed_fig1` to diagnose atmospheric CO2 contributions from land and ocean to plot ensemble timeseries as Figure 1a-c;
    then, run `fig_2` based on the produced results to create Figure 2.
-   `fig_3_4` calculates and then plots Figures 3 & 4.
-   `diagnosed_fig1` also plots Figure 1d+e and SI ensemble timeseries figures.
-   `SI` create various figures for the Supplementary information.

## Missing

-   Model output aggregation into `ds`-file
-   Comparison predictability horizon definitions
-   Diagnosed verification with prognostic timeseries
-   Mauna Loa RMSE-ACC plot including Betts et al. 2016
