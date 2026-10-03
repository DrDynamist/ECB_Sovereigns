from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
YIELDS = ROOT / "data" / "input" / "sovereign_yields_monthly.csv"
SHOCKS = ROOT / "data" / "input" / "ecb_shocks_monthly.csv"
OUTPUT = ROOT / "data" / "derived" / "analysis_panel.csv"


def main() -> None:
    yields = pd.read_csv(YIELDS, parse_dates=["month", "observation_date"])
    shocks = pd.read_csv(SHOCKS, parse_dates=["month"])

    if "Greece" in set(yields["country"]):
        raise ValueError("Greece must not appear in the baseline sample.")
    if yields.duplicated(["country", "maturity_years", "month"]).any():
        raise ValueError("Duplicate country-maturity-month rows in yield data.")
    if shocks.duplicated(["month"]).any():
        raise ValueError("Duplicate months in shock data.")
    if set(yields["maturity_years"].unique()) - {2, 5, 10}:
        raise ValueError("Unexpected maturity outside 2Y, 5Y, and 10Y.")

    # A left merge retains post-October-2025 yields. They are needed to form
    # future outcomes for the final shock observations.
    panel = yields.merge(shocks, on="month", how="left", validate="many_to_one")
    panel["shock_sample"] = panel["mp_median_bps"].notna()
    panel = panel.sort_values(["country", "maturity_years", "month"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")

    shock_rows = panel["shock_sample"].sum()
    print(f"Wrote {len(panel):,} rows to {OUTPUT}")
    print(f"Rows in the January 1999-October 2025 shock window: {shock_rows:,}")
    print(f"Countries: {', '.join(sorted(panel['country'].unique()))}")


if __name__ == "__main__":
    main()
