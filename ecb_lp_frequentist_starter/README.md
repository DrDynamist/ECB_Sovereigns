# ECB shocks and sovereign spreads: frequentist starter

This is the simplified baseline project. Greece is excluded. Excel is not used for estimation.

## Research question

Do ECB monetary-policy shocks and central-bank-information shocks affect euro-area sovereign spreads differently across countries and across the 2Y, 5Y, and 10Y maturities?

The baseline does not impose a core-periphery classification. It estimates a response for every country and then tests whether those responses are statistically equal.

## Files

```text
data/input/sovereign_yields_monthly.csv   Audited monthly yields and German spreads
data/input/ecb_shocks_monthly.csv         Jarocinski-Karadi monthly shock series
data/derived/analysis_panel.csv           Reproducible merge used by the regressions
code/build_analysis_panel.py              Creates the analysis panel
code/run_local_projections.py             Estimates the frequentist local projections
results/lp_country_coefficients.csv       Country response estimates and confidence intervals
results/pooling_tests.csv                 Tests of whether country responses can be pooled
```

## What one row means

One row of `analysis_panel.csv` is one country, one maturity, and one calendar month. For example, a row may represent Italy's 5Y yield in March 2019, the matched German 5Y yield, their spread, and that month's two ECB shocks.

The panel remains in long format because country and maturity are variables. This makes grouping, interactions, fixed effects, and missing-data checks explicit in code. It is not a separate statistical model.

## Outcome constructed by the code

For each horizon `h = 0,...,12`, the script constructs

```text
spread_response(i,m,t,h) = spread(i,m,t+h) - spread(i,m,t-1)
```

This is the cumulative change in country `i`'s matched-maturity sovereign spread from the month before the shock through horizon `h`.

The future-response columns are constructed in memory. They do not clutter the permanent CSV.

## First frequentist specification

The code estimates each maturity and horizon separately. It includes:

- a separate intercept for every country;
- a separate monetary-policy-shock coefficient for every country;
- a separate information-shock coefficient for every country;
- three lags of monthly spread changes;
- three lags of both ECB shocks.

The covariance estimator aggregates regression scores by calendar month and applies Newey-West weights. This accommodates common shocks across countries and serial correlation from overlapping local-projection outcomes.

The coefficient unit is intuitive: basis points of spread response following a one-basis-point ECB shock.

## What pooling means

For each maturity, horizon, and shock, `pooling_tests.csv` tests

```text
H0: France response = Italy response = Spain response = ...
```

- A large p-value means the data do not reject a common response. Complete pooling may be reasonable.
- A small p-value means at least one country response differs. Complete pooling is too restrictive.
- The later regularized model will sit between these cases by shrinking country coefficients toward one another without forcing equality.

These tests are the frequentist starting point. They should be understood before adding penalties, distance-weighted pooling, or Bayesian priors.

## Run the project

From this directory:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python code/build_analysis_panel.py
python code/run_local_projections.py
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python code/build_analysis_panel.py
python code/run_local_projections.py
```

## Important sample detail

The shock file runs from January 1999 through October 2025. The yield input continues through September 2026 because later yield observations are needed to construct future outcomes. Consequently, October 2025 can be used through horizon 11, while September 2025 is the final shock month with a complete horizon-12 response.

Belgium's supplied middle-maturity series was 15Y rather than 5Y, so Belgium 5Y is absent. Finland 5Y begins later than its 2Y and 10Y series. The code uses available observations and never fills these gaps artificially.
