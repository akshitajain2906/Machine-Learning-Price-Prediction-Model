# Machine Learning Price Prediction Model

Capstone project completed as part of a course.

## Overview

How much of an apartment's price comes down to neighborhood alone, independent of size? Can a regularized linear model beat a naive guess by enough to be useful, and does it hold up on listings it's never seen? This project builds a Ridge regression pipeline to answer those questions, predicting apartment price from surface area, coordinates, and neighborhood.

The analysis covers:

- 6,507 cleaned apartment listings across 57 neighborhoods
- Listings filtered to a single metro area, under $400,000 USD
- 4 features after cleaning: surface area, latitude, longitude, neighborhood
- An 80/20 train/test split (5,205 / 1,302 listings), added independently for honest evaluation

## Tools & Technologies

Python · pandas · scikit-learn · category_encoders · matplotlib · seaborn

## Business Questions

- How much does location alone explain in price, independent of apartment size?
- Can a regularized linear model meaningfully beat a naive mean-price baseline?
- Which neighborhoods carry the strongest positive and negative price premiums?
- Does the model generalize to unseen listings, or does it overfit to the training set?

## Key Findings

**Model Accuracy**

| Metric | Value |
|---|---|
| Baseline MAE (predicting the mean for every listing) | $44,845.54 |
| Training MAE | $24,172.47 |
| Test MAE | $25,360.04 |
| Improvement over baseline | ~43% |

The model cuts prediction error nearly in half versus simply guessing the average price for every listing. Training and test MAE sit within ~$1,200 of each other, which means the regularization is doing its job - the model isn't memorizing the training set, it's actually learning a pattern that holds on data it hasn't seen.

**Neighborhood Price Premiums**

Strongest positive effect: Puerto Madero (+$120,552), Las Cañitas (+$78,188), Recoleta (+$49,794) - all well known as some of the most expensive areas in the city. Strongest negative effect: Villa Lugano (–$55,814), Villa Soldati (–$46,156), Boca (–$39,303) - consistent with how these areas are generally perceived in the local market.

The spread between the highest and lowest neighborhood coefficients is over $176,000 - larger than the average apartment price in the entire dataset. Location alone moves price more than size does: doubling an apartment's surface area adds less to predicted price than the gap between a premium and a discounted neighborhood.

**Size Effect**

Holding location constant, each additional square meter of covered surface adds roughly $2,206 to predicted price — a much smaller and more linear effect than neighborhood, which behaves more like a step function than a gradient.

## Methodology

Four categories of columns were dropped, each for a distinct reason rather than a blanket cleanup: high null counts (`expenses`, `floor`), unusably low or high cardinality (`operation`, `currency`, `property_type`, `properati_url`), target leakage (`price`, `price_aprox_local_currency`, `price_per_m2`, `price_usd_per_m2` are all derived from the value being predicted), and multicollinearity (`surface_total_in_m2` and `rooms` correlate with `surface_covered_in_m2` at 0.73 and 0.78 respectively, adding redundancy without new signal).

```python
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(strategy="median"),
    Ridge(alpha=1)
)
```

Ridge was chosen over plain linear regression because one-hot encoding 57 neighborhoods creates a high-dimensional, sparse feature space where ordinary least squares tends to latch onto noise. L2 regularization shrinks those coefficients toward zero without dropping any neighborhood entirely, which gives better generalization without needing to manually pick which neighborhoods matter. The fitted pipeline is wrapped in a `make_prediction()` function so a price estimate can be generated from raw inputs without manually constructing a DataFrame.

## Business Implications

For a brokerage or listing platform, this model quantifies something agents normally only state qualitatively - that neighborhood reputation translates into a specific dollar premium, not just a vague "more desirable" label. A $176,000 swing between the strongest and weakest neighborhoods means listings could be flagged as mispriced relative to comparable units elsewhere in the same area, surfacing either pricing errors or acquisition opportunities before a property goes to market. Because the location effect dominates the size effect this strongly, any pricing or investment strategy built on this data should weight neighborhood far more heavily than square footage alone.
