from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from research_lib import OUTPUT, PROCESSED
from run_main import make_sample, setup_logging


def randomization_inference(sample: pd.DataFrame, draws: int = 1000, seed: int = 20261029) -> dict:
    balanced = sample.dropna(subset=["ln_price"]).copy()
    counts = balanced.groupby("citycode")["year"].transform("size")
    balanced = balanced.loc[counts.eq(13)].copy()
    wide_y = balanced.pivot(index="citycode", columns="year", values="ln_price").sort_index()
    years = wide_y.columns.to_numpy()
    y_matrix = wide_y.to_numpy(dtype=float)
    y_residual = (
        y_matrix
        - y_matrix.mean(axis=1, keepdims=True)
        - y_matrix.mean(axis=0, keepdims=True)
        + y_matrix.mean()
    )

    def beta_from_codes(codes: set[int]) -> float:
        d_matrix = np.zeros_like(y_matrix)
        for code in codes:
            d_matrix[wide_y.index.get_loc(code), years >= 2014] = 1.0
        d_residual = (
            d_matrix
            - d_matrix.mean(axis=1, keepdims=True)
            - d_matrix.mean(axis=0, keepdims=True)
            + d_matrix.mean()
        )
        return float((y_residual * d_residual).sum() / (d_residual**2).sum())

    eligible_codes = wide_y.index.to_numpy()
    treated_count = int(
        balanced.loc[balanced["treated_main"].eq(1), "citycode"].nunique()
    )
    observed_treated = set(
        balanced.loc[balanced["treated_main"].eq(1), "citycode"].astype(int)
    )
    observed = beta_from_codes(observed_treated)
    rng = np.random.default_rng(seed)
    estimates = []
    for draw in range(draws):
        pseudo = rng.choice(eligible_codes, size=treated_count, replace=False)
        estimates.append(beta_from_codes(set(int(code) for code in pseudo)))
        if (draw + 1) % 100 == 0:
            logging.info("Randomization draw %s/%s", draw + 1, draws)
    estimates = np.asarray(estimates)
    two_sided_p = (1 + np.sum(np.abs(estimates) >= abs(observed))) / (draws + 1)
    upper_p = (1 + np.sum(estimates >= observed)) / (draws + 1)
    return {
        "observed_coef": float(observed),
        "draws": draws,
        "seed": seed,
        "pseudo_coef_mean": float(estimates.mean()),
        "pseudo_coef_sd": float(estimates.std(ddof=1)),
        "two_sided_p": float(two_sided_p),
        "one_sided_upper_p": float(upper_p),
        "pseudo_quantiles": {
            str(q): float(np.quantile(estimates, q))
            for q in [0.01, 0.05, 0.5, 0.95, 0.99]
        },
        "observed_treated_codes": sorted(observed_treated),
        "draw_values": estimates.tolist(),
    }


def rolling_preperiod_placebos(sample: pd.DataFrame) -> pd.DataFrame:
    pre = sample.loc[sample["year"].between(2007, 2013)].dropna(
        subset=["ln_price"]
    ).copy()
    counts = pre.groupby("citycode")["year"].transform("size")
    pre = pre.loc[counts.eq(7)]
    wide = pre.pivot(index="citycode", columns="year", values="ln_price").sort_index()
    years = wide.columns.to_numpy()
    y = wide.to_numpy(dtype=float)
    y_residual = (
        y
        - y.mean(axis=1, keepdims=True)
        - y.mean(axis=0, keepdims=True)
        + y.mean()
    )
    treated_codes = (
        pre.loc[pre["treated_main"].eq(1), "citycode"].drop_duplicates().tolist()
    )
    rows = []
    for fake_year in range(2008, 2014):
        d = np.zeros_like(y)
        for code in treated_codes:
            d[wide.index.get_loc(code), years >= fake_year] = 1.0
        d_residual = (
            d
            - d.mean(axis=1, keepdims=True)
            - d.mean(axis=0, keepdims=True)
            + d.mean()
        )
        beta = float((y_residual * d_residual).sum() / (d_residual**2).sum())
        residual = y_residual - beta * d_residual
        meat = np.sum((d_residual * residual).sum(axis=1) ** 2)
        g = d.shape[0]
        se = float(
            np.sqrt((g / (g - 1)) * meat / (d_residual**2).sum() ** 2)
        )
        rows.append(
            {
                "fake_treatment_year": fake_year,
                "coef": beta,
                "se": se,
                "ci_low": beta - 1.96 * se,
                "ci_high": beta + 1.96 * se,
                "pvalue": 2 * (1 - stats.norm.cdf(abs(beta / se))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    panel = pd.read_csv(PROCESSED / "panel_city_year.csv", low_memory=False)
    sample = make_sample(panel)
    pre = sample.loc[sample["year"].le(2013)].copy()
    pre = pre.dropna(subset=["ln_price"])
    counts = pre.groupby("citycode")["year"].transform("size")
    pre = pre.loc[counts.eq(7)]
    wide = pre.pivot(index="citycode", columns="year", values="ln_price").sort_index()
    y_matrix = wide.to_numpy(dtype=float)
    y_residual = (
        y_matrix
        - y_matrix.mean(axis=1, keepdims=True)
        - y_matrix.mean(axis=0, keepdims=True)
        + y_matrix.mean()
    )
    treated_codes = pre.loc[pre["treated_main"].eq(1), "citycode"].unique()
    d_matrix = np.zeros_like(y_matrix)
    for code in treated_codes:
        d_matrix[wide.index.get_loc(code), wide.columns >= 2010] = 1
    d_residual = (
        d_matrix
        - d_matrix.mean(axis=1, keepdims=True)
        - d_matrix.mean(axis=0, keepdims=True)
        + d_matrix.mean()
    )
    fake_beta = float((y_residual * d_residual).sum() / (d_residual**2).sum())
    # Cluster-robust SE for the balanced closed-form estimator.
    residual = y_residual - fake_beta * d_residual
    meat = np.sum((d_residual * residual).sum(axis=1) ** 2)
    g = d_matrix.shape[0]
    correction = g / (g - 1)
    fake_se = float(np.sqrt(correction * meat / (d_residual**2).sum() ** 2))
    fake = {"coef": fake_beta, "se": fake_se}

    random = randomization_inference(sample, draws=1000)
    pd.DataFrame(
        {
            "draw": np.arange(1, len(random["draw_values"]) + 1),
            "coef": random["draw_values"],
        }
    ).to_csv(OUTPUT / "placebo_draws.csv", index=False, encoding="utf-8-sig")
    random_for_json = {
        key: value for key, value in random.items() if key != "draw_values"
    }
    result = {
        "fake_treatment_2010": {
            **fake,
            "sample_years": [2007, 2013],
            "interpretation": (
                "A non-zero placebo indicates pre-existing differential paths."
            ),
        },
        "randomization_inference": random_for_json,
    }
    (OUTPUT / "placebo_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Fake 2010 treatment beta=%.4f SE=%.4f", fake["coef"], fake["se"])
    logging.info(
        "Randomization two-sided p=%.4f", random["two_sided_p"]
    )
    rolling = rolling_preperiod_placebos(sample)
    rolling.to_csv(
        OUTPUT / "preperiod_placebo_years.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
