# Spring and Ilyina, 2019, manuscript in prep.

The paper is still in the process of writing and has not been submitted yet.

## Aim

This repo is setup for scientists interested to reproduce our `Spring and Ilyina, 2019` paper. It contains scripts to reproduce the analysis and create the shown figures. It is inspired by `Irving (2015)` to enhance reproducibility in geosciences.

Irving, Damien. “A Minimum Standard for Publishing Computational Results in the Weather and Climate Sciences.” Bulletin of the American Meteorological Society 97, no. 7 (October 7, 2015): 1149–58. <https://doi.org/10/gf4wzh>.

## Packages used mostly

-   model output aggregation: `cdo`
-   analysis: `xarray`
-   visualisation: `matplotlib`, `cartopy`
-   predictive skill analysis: [`climpred`](https://climpred.readthedocs.io/)
-   (private repo) plotting routines and data storage on supercomputer: `PMMPIESM`

## Environment

Dependencies (Packages installed) can be found in `requirements.txt`.
Installed via conda (see setup `conda_info.txt`) and pip.
