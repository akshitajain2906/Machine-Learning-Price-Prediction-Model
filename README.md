# Predicting Apartment Prices in Buenos Aires

Capstone project completed as part of a course. A regularized linear regression model that predicts apartment prices in Buenos Aires (Capital Federal) from size, location, and neighborhood.

## Overview

Buenos Aires has one of Latin America's most active real estate markets, and price is driven heavily by neighborhood as much as by size. This project builds a Ridge regression pipeline that predicts apartment price from surface area, coordinates, and neighborhood, with every cleaning decision justified by what the data actually showed rather than applied by default.

## Tools

Python · pandas · scikit-learn · category_encoders · matplotlib · seaborn

## Dataset

Real estate listings for Buenos Aires, filtered down to apartments in Capital Federal priced under $400,000 USD. After cleaning, the working dataset contains 6,507 listings across 57 neighborhoods.

## Methodology

**Data cleaning.** A single `wrangle()` function handles every transformation: subsetting to Capital Federal apartments under $400K, trimming the top and bottom 10% of `surface_covered_in_m2` to remove extreme outliers, splitting the combined `lat-lon` string into usable numeric columns, and extracting neighborhood from a nested location string.

**Feature selection.** Columns were dropped for four distinct reasons rather than a blanket cleanup: high null counts (`expenses`, `floor`), unusably low or high cardinality (`operation`, `currency`, `property_type`, `properati_url`), target leakage (`price`, `price_aprox_local_currency`, `price_per_m2`, `price_usd_per_m2` are all derived from the price we're predicting), and multicollinearity (`surface_total_in_m2` and `rooms` correlate with `surface_covered_in_m2` at 0.73 and 0.78 respectively, adding redundancy without new signal).

**Train/test split.** An 80/20 split (5,205 / 1,302 listings) was added to hold out unseen data for honest evaluation.

**Baseline.** Before modeling, predicting the mean price for every listing gives a baseline MAE of $44,845.54 — the floor any real model needs to beat.

**Model.** A pipeline of `OneHotEncoder` (neighborhood), `SimpleImputer` (median, for the ~4% of listings missing coordinates), and `Ridge` regression. Ridge was chosen over plain linear regression because one-hot encoding 57 neighborhoods creates a high-dimensional, sparse feature space where ordinary least squares tends to overfit; L2 regularization shrinks those coefficients toward zero without dropping any neighborhood entirely.

## Results

| Metric | Value |
|---|---|
| Baseline MAE | $44,845.54 |
| Training MAE | $24,172.47 |
| Test MAE | $25,360.04 |
| Improvement over baseline | ~43% |

Training and test MAE sit within ~$1,200 of each other, with no sign of overfitting, and both land well below the baseline.

## Model Interpretation

Looking at the neighborhood coefficients (excluding `lat`/`lon`, which are continuous and dominate by scale):

- **Strongest positive effect:** Puerto Madero (+$120,552), Las Cañitas (+$78,188), and Recoleta (+$49,794) — all well known as some of the most expensive areas in the city.
- **Strongest negative effect:** Villa Lugano (–$55,814), Villa Soldati (–$46,156), and Boca (–$39,303) — consistent with how these areas are generally perceived in the local market.
- **Size:** holding location constant, each additional square meter of covered surface adds roughly $2,206 to predicted price.

## Prediction Function

The fitted pipeline is wrapped in `make_prediction(area, lat, lon, neighborhood)` so a price estimate can be generated from raw inputs without manually constructing a DataFrame.
