# College Football Betting Model

The purpose of this repo is to build and develop a robust college football betting model that consistently predicts +CLV to the opening line and closely predicts the closing line such that betting the opening line on the Sunday lines drop will lead to profitability. This model is used only for spreads, totals, and potentially MLs.  

## Data

The data used in this model comes from the collegefootballdata.com API. Data used for training starts in 2013 and runs through the present. Returning production data is only available starting in 2014 (returning production from 2013 season). Opening line data is only available starting in 2021, so the actual profitability of the models is only tested from 2021 - 2024, whereas the stastical predictive power of the models is measured across the full dataset (2013 - 2024). 

## Workflow

The data is stored in a sqlite database, so the `datapull.ipynb` notebook is run to call the API for the various data used and added to the `cfb_data.db` you will create locally and in the same directory as this repo. 

`elo_model.py` gives the general framework for the ELO based power rankings model used. 

`elo_optimization.py` is run to perform an Optuna multivariate optimization of the various tunable parameters used in this model in order to get the strongest predictive power possible. 

`profit_loss_sim.py` then takes the optimized parameters and determines the actual profit and loss of the betting strategy (use predicted spread vs. opening line to make bets).
