# Scripts

This repo was setup for scientists interested to reproduce our `Spring and Ilyina, 2019` paper. Please find the used scripts here.

## How to use

-   Setup the `conda` environment as described by `requirements.txt`
-   Use the jupyter notebooks (`Spring_and_Ilyina_2019.ipynb`) to inspect results interactively; otherwise run `python *.py`. Note that the jupyter notebook was not updated with the changes due to revisions.
-   First, run `diagnosed_fig1` to diagnose atmospheric CO2 contributions from land and ocean to plot ensemble timeseries as Figure 3c;
    then, run `fig_1` based on the produced results to create Figure 1.
-   `fig_2_4` calculates and then plots Figures 2 & 4.
-   `diagnosed_fig3` also plots Figure 3a,b.
-   `SI` create various figures for the Supplementary information.
-   reproduced results statistical model `Betts et al. 2016, 2018` in python `Rebuilding_Betts_2016.ipynb`

## reviewer_comments

This notebook was created to create additional supplementary figures asked for by reviewer #2 to show how the variability of MPI-ESM compares to observation-based products.

We use the following observation-based datasets to compare to MPI-ESM:

-   `variable`: `name` :`dataset`: `link`
-   `co2_flx_ocean`: ocean CO2 flux :`SOM-FFN` <https://www.mpimet.mpg.de/mitarbeiter/peter-landschuetzer/links/>
-   `co2_flx_land`: terrestrial CO2 flux : `Jena CarboScope CO2 inversion sEXTocNEET_v4.3` <https://www.bgc-jena.mpg.de/CarboScope/?ID=s>
-   `CO2`: surface atmospheric CO2 mixing-ratio: `Jena CarboScope CO2 inversion sEXTocNEET_v4.3` <https://www.bgc-jena.mpg.de/CarboScope/?ID=s>

## Not documented

-   Model output aggregation into `ds`-file, done with `cdo` and `xarray` on DKRZ supercomputer `mistral`
