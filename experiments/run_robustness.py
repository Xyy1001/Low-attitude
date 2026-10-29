from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research_lib import (
    ALT_AREA_PILOTS,
    ALT_ISLAND_PILOTS,
    CORE_PILOTS,
    OUTPUT,
    PROCESSED,
    add_spatial_treatment,
    event_study,
    fit_fe,
    ihs,
)
from run_main import make_sample, setup_logging, tidy_result


def matched_sample(sample: pd.DataFrame, neighbors_per_treated: int = 5) -> pd.DataFrame:
    cols = [
        "ln_gdp_base",
        "ln_pop_base",
        "secondary_share_base",
        "airport_exposure_c",
    ]
    city = sample.groupby("citycode", as_index=False).agg(
        treated=("treated_main", "max"),
        **{col: (col, "first") for col in cols},
    )
    city = city.dropna(subset=cols)
    matrix = city[cols].to_numpy(dtype=float)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
    treated_idx = np.flatnonzero(city["treated"].to_numpy() == 1)
    control_idx = np.flatnonzero(city["treated"].to_numpy() == 0)
    selected_controls = set()
    for idx in treated_idx:
        distance = np.sqrt(((matrix[control_idx] - matrix[idx]) ** 2).sum(axis=1))
        nearest = control_idx[np.argsort(distance)[:neighbors_per_treated]]
        selected_controls.update(int(value) for value in nearest)
    codes = set(city.iloc[treated_idx]["citycode"].astype(int))
    codes.update(int(city.iloc[list(selected_controls)]["citycode"].astype(int).iloc[i]) for i in range(len(selected_controls)))
    return sample.loc[sample["citycode"].isin(codes)].copy()


def alternative_definition(panel: pd.DataFrame, mode: str) -> pd.DataFrame:
    data = panel.loc[panel["year"].between(2007, 2019)].copy()
    if mode == "broad_2014":
        codes = set(CORE_PILOTS) | set(ALT_AREA_PILOTS) | set(ALT_ISLAND_PILOTS)
        keep = set(data["citycode"])
    elif mode == "hainan_only":
        codes = set(ALT_ISLAND_PILOTS)
        keep = codes | {
            int(code)
            for code in data.loc[
                ~data["citycode"].isin(
                    set(CORE_PILOTS) | set(ALT_AREA_PILOTS) | set(ALT_ISLAND_PILOTS)
                ),
                "citycode",
            ].unique()
        }
    elif mode == "shenyang_alt":
        codes = set(ALT_AREA_PILOTS)
        keep = codes | {
            int(code)
            for code in data.loc[
                ~data["citycode"].isin(
                    set(CORE_PILOTS) | set(ALT_AREA_PILOTS) | set(ALT_ISLAND_PILOTS)
                ),
                "citycode",
            ].unique()
        }
    else:
        raise ValueError(mode)
    data = data.loc[data["citycode"].isin(keep)].copy()
    data["treated_alt"] = data["citycode"].isin(codes).astype(int)
    data["post_alt"] = (
        data["treated_alt"].eq(1) & data["year"].ge(2014)
    ).astype(int)
    data["ln_gdp_base_trend"] = data["ln_gdp_base"] * (data["year"] - 2007)
    return data


def synthetic_gap(sample: pd.DataFrame, outcome: str = "ln_price") -> dict:
    city_year = sample.pivot(index="year", columns="citycode", values=outcome)
    treated = (
        sample.loc[sample["treated_main"].eq(1)]
        .groupby("year")[outcome]
        .mean()
        .reindex(city_year.index)
    )
    controls = city_year.loc[:, city_year.notna().all(axis=0)].copy()
    control_codes = [code for code in controls.columns if code not in CORE_PILOTS]
    controls = controls.loc[:, control_codes]
    covariates = [
        "ln_gdp_base",
        "ln_pop_base",
        "secondary_share_base",
        "airport_exposure_c",
    ]
    city_cov = sample.groupby("citycode")[covariates].first().dropna()
    treated_mean = city_cov.loc[
        city_cov.index.isin(CORE_PILOTS), covariates
    ].mean()
    donor_codes = [code for code in control_codes if code in city_cov.index]
    scaled = city_cov.loc[donor_codes, covariates].copy()
    scale = scaled.std(ddof=0).replace(0, 1)
    distance = np.sqrt(
        (((scaled - treated_mean) / scale) ** 2).sum(axis=1)
    ).sort_values()
    donor_codes = list(distance.index[:30])
    controls = controls.loc[:, donor_codes]
    pre = controls.index <= 2013
    post = controls.index >= 2014
    x = controls.to_numpy(dtype=float)
    y = treated.to_numpy(dtype=float)

    def objective(w: np.ndarray) -> float:
        gap = y[pre] - x[pre] @ w
        return float(np.mean(gap**2) + 0.0005 * np.sum(w**2))

    n = x.shape[1]
    initial = np.ones(n) / n
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    weights = result.x
    synthetic = pd.Series(x @ weights, index=controls.index)
    gap = treated - synthetic
    return {
        "outcome": outcome,
        "pre_rmse": float(np.sqrt(np.mean(gap.loc[pre] ** 2))),
        "post_average_gap": float(gap.loc[post].mean()),
        "pre_average_gap": float(gap.loc[pre].mean()),
        "did_gap": float(gap.loc[post].mean() - gap.loc[pre].mean()),
        "effective_controls": int((weights > 1e-4).sum()),
        "max_weight": float(weights.max()),
        "optimizer_success": bool(result.success),
        "path": [
            {
                "year": int(year),
                "treated": float(treated.loc[year]),
                "synthetic": float(synthetic.loc[year]),
                "gap": float(gap.loc[year]),
            }
            for year in controls.index
        ],
        "weights": [
            {"citycode": int(code), "weight": float(weight)}
            for code, weight in zip(controls.columns, weights)
            if weight > 1e-4
        ],
        "interpretation": (
            "Descriptive synthetic-control gap. No inferential p-value is claimed."
        ),
    }


def main() -> None:
    setup_logging()
    panel = pd.read_csv(PROCESSED / "panel_city_year.csv", low_memory=False)
    sample = make_sample(panel)
    rows = []
    extrema = []

    for outcome in ["ln_price", "ln_area", "ln_revenue"]:
        for col, label in [
            (outcome.replace("ln_", "ln_") + "_5_95", "5/95 winsorization"),
        ]:
            if col in sample:
                result = fit_fe(sample, col)
                rows.append(tidy_result(result, label, "TWFE"))

    for outcome in ["price_re_zpg", "area_re_zpg", "revenue_re_zpg"]:
        label = "IHS_" + outcome
        sample[label] = ihs(sample[outcome])
        result = fit_fe(sample, label)
        rows.append(tidy_result(result, "IHS transformation", "TWFE"))

    no_municipality = sample.loc[sample["municipality"].eq(0)].copy()
    rows.append(
        tidy_result(
            fit_fe(no_municipality, "ln_price"),
            "Exclude four municipalities",
            "TWFE",
        )
    )
    rows.append(
        tidy_result(
            fit_fe(sample, "ln_price", province_year_fe=True),
            "Province-year FE",
            "TWFE",
        )
    )
    rows.append(
        tidy_result(
            fit_fe(sample, "ln_price", cluster="province_code"),
            "Cluster by province",
            "TWFE",
        )
    )
    rows.append(
        tidy_result(
            fit_fe(sample.loc[sample["year"].le(2016)], "ln_price"),
            "2007-2016 window",
            "TWFE",
        )
    )

    for mode in ["broad_2014", "hainan_only", "shenyang_alt"]:
        alt = alternative_definition(panel, mode)
        result = fit_fe(
            alt,
            "ln_price",
            treatment="post_alt",
            extra_columns=["ln_gdp_base_trend"],
        )
        rows.append(tidy_result(result, f"Alternative treatment: {mode}", "TWFE"))

    for code in CORE_PILOTS:
        subset = sample.loc[sample["citycode"].ne(code)].copy()
        result = fit_fe(subset, "ln_price")
        extrema.append(
            {
                "omitted_citycode": code,
                "omitted_city": CORE_PILOTS[code],
                "coef": result["coef"],
                "se": result["se"],
                "pvalue": result["pvalue"],
            }
        )
        rows.append(
            tidy_result(
                result,
                f"Leave out {CORE_PILOTS[code]}",
                "TWFE",
            )
        )

    matched = matched_sample(sample)
    matched_result = fit_fe(matched, "ln_price")
    rows.append(tidy_result(matched_result, "Covariate-matched sample", "TWFE"))
    matched_event = event_study(matched, "ln_price")

    for radius in [50, 100, 200, 300]:
        spatial = add_spatial_treatment(sample, radius_km=radius, minimum_neighbors=5)
        if spatial["citycode"].nunique() < 100:
            continue
        result = fit_fe(
            spatial,
            "ln_price",
            extra_columns=["spatial_post"],
            focal_terms=["post", "spatial_post"],
        )
        row = {
            "sample": f"{radius} km inverse-distance spillover",
            "model": "TWFE spatial association",
            "outcome": "ln_price",
            "term": "spatial_post",
            "nobs": result["nobs"],
            "cities": result["cities"],
            "clusters": result["clusters"],
        }
        row.update(result["focal_coefficients"]["spatial_post"])
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "robustness_results.csv", index=False, encoding="utf-8-sig")
    leaveout = pd.DataFrame(extrema)
    leaveout.to_csv(OUTPUT / "leave_one_treated_out.csv", index=False, encoding="utf-8-sig")
    matched_event.to_csv(
        OUTPUT / "event_study_matched_sample.csv", index=False, encoding="utf-8-sig"
    )
    synthetic = synthetic_gap(sample, "ln_price")
    synthetic_path = pd.DataFrame(synthetic.pop("path"))
    synthetic_weights = pd.DataFrame(synthetic.pop("weights"))
    synthetic_path.to_csv(
        OUTPUT / "synthetic_control_path.csv", index=False, encoding="utf-8-sig"
    )
    synthetic_weights.to_csv(
        OUTPUT / "synthetic_control_weights.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "matched_sample_cities": int(matched["citycode"].nunique()),
        "matched_event_pretrend_f": matched_event.attrs["pretrend_f"],
        "matched_event_pretrend_p": matched_event.attrs["pretrend_p"],
        "leave_one_out_coef_range": [
            float(leaveout["coef"].min()),
            float(leaveout["coef"].max()),
        ],
        "synthetic_control": synthetic,
        "robustness_note": (
            "Alternative pilots are coding choices at the regional perimeter, not "
            "new source-verified city-level rollout dates."
        ),
    }
    (OUTPUT / "robustness_results.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Wrote %s robustness rows", len(table))
    logging.info("Matched pretrend p=%.4f", matched_event.attrs["pretrend_p"])
    logging.info("Synthetic-control result: %s", synthetic)


if __name__ == "__main__":
    main()
