"""Frequentist local projections with country-specific shock coefficients.

For each maturity and horizon h=0,...,12, estimate

    spread[i,t+h] - spread[i,t-1]
      = country intercepts
      + country-specific MP and CBI shock coefficients
      + three lags of spread changes and both shocks
      + error.

Inference uses a Newey-West covariance matrix built from calendar-month score
sums. This allows arbitrary contemporaneous dependence across countries and
serial dependence across months. The output includes Wald tests of whether the
country-specific shock coefficients are equal.

Run from the project root:
    python code/run_local_projections.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "derived" / "analysis_panel.csv"
RESULTS = ROOT / "results"
HORIZONS = range(13)
LAGS = 3


def calendar_shift_lookup(panel: pd.DataFrame, frame: pd.DataFrame, months: int, column: str) -> np.ndarray:
    """Look up an exact calendar-month shift within a country-maturity series."""
    index = panel.set_index(["country", "maturity_years", "month"])[column]
    keys = pd.MultiIndex.from_arrays(
        [
            frame["country"],
            frame["maturity_years"],
            frame["month"] + pd.offsets.MonthBegin(months),
        ],
        names=["country", "maturity_years", "month"],
    )
    return index.reindex(keys).to_numpy(dtype=float)


def newey_west_by_time(X: np.ndarray, residuals: np.ndarray, dates: pd.Series, bandwidth: int) -> np.ndarray:
    """HAC covariance using cross-sectional score sums for each calendar month."""
    unique_dates, inverse = np.unique(dates.to_numpy(), return_inverse=True)
    scores = np.zeros((len(unique_dates), X.shape[1]))
    np.add.at(scores, inverse, X * residuals[:, None])

    meat = scores.T @ scores
    for lag in range(1, min(bandwidth, len(unique_dates) - 1) + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)

    bread = np.linalg.pinv(X.T @ X)
    correction = len(unique_dates) / max(len(unique_dates) - 1, 1)
    return correction * bread @ meat @ bread


def wald_equal(beta: np.ndarray, covariance: np.ndarray, coefficient_indices: list[int]) -> tuple[float, int, float]:
    """Test equality of a set of coefficients against the first coefficient."""
    k = len(beta)
    restrictions = np.zeros((len(coefficient_indices) - 1, k))
    first = coefficient_indices[0]
    for row, current in enumerate(coefficient_indices[1:]):
        restrictions[row, current] = 1.0
        restrictions[row, first] = -1.0
    difference = restrictions @ beta
    variance = restrictions @ covariance @ restrictions.T
    statistic = float(difference.T @ np.linalg.pinv(variance) @ difference)
    degrees = int(np.linalg.matrix_rank(restrictions))
    return statistic, degrees, float(chi2.sf(statistic, degrees))


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["month", "observation_date"])
    panel = panel.sort_values(["country", "maturity_years", "month"]).copy()
    shock_month = panel.drop_duplicates("month").set_index("month")

    panel["spread_lag1_bps"] = calendar_shift_lookup(panel, panel, -1, "spread_bps")
    for lag in range(1, LAGS + 1):
        later = calendar_shift_lookup(panel, panel, -lag, "spread_bps")
        earlier = calendar_shift_lookup(panel, panel, -(lag + 1), "spread_bps")
        panel[f"dspread_lag{lag}_bps"] = later - earlier

        shifted_month = panel["month"] + pd.offsets.MonthBegin(-lag)
        for shock in ["mp_median_bps", "cbi_median_bps"]:
            panel[f"{shock}_lag{lag}"] = shock_month[shock].reindex(shifted_month).to_numpy()
    return panel


def build_design(sample: pd.DataFrame, countries: list[str]) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    columns, names = [], []

    # Country intercepts.
    for country in countries:
        columns.append((sample["country"] == country).astype(float).to_numpy())
        names.append(f"intercept_{country}")

    shock_indices: dict[str, list[int]] = {}
    for shock in ["mp_median_bps", "cbi_median_bps"]:
        shock_indices[shock] = []
        for country in countries:
            shock_indices[shock].append(len(columns))
            columns.append(
                ((sample["country"] == country).astype(float) * sample[shock]).to_numpy()
            )
            names.append(f"{shock}_{country}")

    controls = [f"dspread_lag{x}_bps" for x in range(1, LAGS + 1)]
    controls += [f"{shock}_lag{x}" for shock in ["mp_median_bps", "cbi_median_bps"] for x in range(1, LAGS + 1)]
    for control in controls:
        columns.append(sample[control].to_numpy(dtype=float))
        names.append(control)

    return np.column_stack(columns), names, shock_indices


def main() -> None:
    panel = prepare_panel()
    countries = sorted(panel["country"].unique())
    coefficients, tests = [], []

    for maturity in [2, 5, 10]:
        maturity_panel = panel[panel["maturity_years"] == maturity].copy()
        for horizon in HORIZONS:
            future_spread = calendar_shift_lookup(panel, maturity_panel, horizon, "spread_bps")
            maturity_panel["response"] = future_spread - maturity_panel["spread_lag1_bps"]

            required = ["response", "mp_median_bps", "cbi_median_bps"]
            required += [f"dspread_lag{x}_bps" for x in range(1, LAGS + 1)]
            required += [f"{shock}_lag{x}" for shock in ["mp_median_bps", "cbi_median_bps"] for x in range(1, LAGS + 1)]
            sample = maturity_panel[maturity_panel["shock_sample"]].dropna(subset=required).copy()

            present = [c for c in countries if c in set(sample["country"])]
            X, names, shock_indices = build_design(sample, present)
            y = sample["response"].to_numpy(dtype=float)
            beta = np.linalg.pinv(X.T @ X) @ X.T @ y
            residuals = y - X @ beta
            covariance = newey_west_by_time(X, residuals, sample["month"], bandwidth=max(horizon + 1, LAGS))
            standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

            for shock in ["mp_median_bps", "cbi_median_bps"]:
                label = "MP" if shock.startswith("mp_") else "CBI"
                for country, index in zip(present, shock_indices[shock]):
                    estimate, se = float(beta[index]), float(standard_errors[index])
                    coefficients.append({
                        "maturity_years": maturity,
                        "horizon_months": horizon,
                        "country": country,
                        "shock": label,
                        "estimate_bps_per_1bp_shock": estimate,
                        "standard_error": se,
                        "ci95_low": estimate - 1.96 * se,
                        "ci95_high": estimate + 1.96 * se,
                        "observations": len(sample),
                        "calendar_months": sample["month"].nunique(),
                    })

                stat, df, pvalue = wald_equal(beta, covariance, shock_indices[shock])
                tests.append({
                    "maturity_years": maturity,
                    "horizon_months": horizon,
                    "restriction": f"equal_{label}_coefficients_across_countries",
                    "wald_chi2": stat,
                    "degrees_of_freedom": df,
                    "p_value": pvalue,
                    "reject_equal_coefficients_at_5pct": pvalue < 0.05,
                    "observations": len(sample),
                    "calendar_months": sample["month"].nunique(),
                })

            # Joint equality restriction for both shock families.
            all_indices = shock_indices["mp_median_bps"] + shock_indices["cbi_median_bps"]
            n = len(present)
            restrictions = []
            for offset in [0, n]:
                base = all_indices[offset]
                for current in all_indices[offset + 1: offset + n]:
                    row = np.zeros(len(beta)); row[current] = 1.0; row[base] = -1.0
                    restrictions.append(row)
            R = np.vstack(restrictions)
            difference = R @ beta
            variance = R @ covariance @ R.T
            stat = float(difference.T @ np.linalg.pinv(variance) @ difference)
            df = int(np.linalg.matrix_rank(R))
            pvalue = float(chi2.sf(stat, df))
            tests.append({
                "maturity_years": maturity,
                "horizon_months": horizon,
                "restriction": "equal_MP_and_CBI_coefficients_across_countries",
                "wald_chi2": stat,
                "degrees_of_freedom": df,
                "p_value": pvalue,
                "reject_equal_coefficients_at_5pct": pvalue < 0.05,
                "observations": len(sample),
                "calendar_months": sample["month"].nunique(),
            })

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coefficients).to_csv(RESULTS / "lp_country_coefficients.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "pooling_tests.csv", index=False)
    print(f"Wrote {len(coefficients):,} coefficient estimates and {len(tests):,} pooling tests to {RESULTS}")


if __name__ == "__main__":
    main()
