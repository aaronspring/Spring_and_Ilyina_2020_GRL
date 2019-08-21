# Scripts

This repo was setup for scientists interested to reproduce our `Spring and Ilyina, 2019` paper. Please find the used scripts here.

## How to use

-   Setup the conda environment as described by `requirements.txt`
-   Use the jupyter notebooks (`Spring_and_Ilyina_2019.ipynb`) to inspect results interactively; otherwise run `python *.py`
-   First, run `diagnosed_fig1` to diagnose atmospheric CO2 contributions from land and ocean to plot ensemble timeseries as Figure 3c;
    then, run `fig_1` based on the produced results to create Figure 1.
-   `fig_2_4` calculates and then plots Figures 2 & 4.
-   `diagnosed_fig3` also plots Figure 3a,b.
-   `SI` create various figures for the Supplementary information.
-   reproduced results statistical model `Betts et al. 2016, 2018` in python `Rebuilding_Betts_2016.ipynb`

## Not documented

-   Model output aggregation into `ds`-file, done with `cdo` and `xarray` on DKRZ supercomputer `mistral`
