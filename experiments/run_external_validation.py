from __future__ import annotations

import json

import pandas as pd

from research_lib import OUTPUT, REAL


def main() -> None:
    data = pd.read_csv(REAL / "drone_delivery" / "Residential_Analysis.csv")
    included = data.loc[data["Include in Analysis?"].fillna(0).eq(1)].copy()
    market_col = "Serviceable Drone Market, 2027"
    trips_col = "Annual Vehicle Trips Replaced by Drone Delivery, 2027"
    time_col = "Annual Travel Time Replaced by Drone Delivery, 2027 (Minutes)"
    distance_col = "One-Way Travel Distance to Nearest Pharmacy (miles)"
    road_time_col = "One-Way Travel Time to Nearest Pharmacy (minutes)"
    residents_col = "Total Residents/\nResidence"
    included["drone_serviceable"] = included[market_col].fillna(0).gt(0)
    grouped = (
        included.groupby("drone_serviceable", as_index=False)
        .agg(
            residences=("Residence ID", "size"),
            median_ground_distance_miles=(distance_col, "median"),
            median_ground_time_minutes=(road_time_col, "median"),
            mean_replaced_travel_time_minutes=(time_col, "mean"),
            total_replaced_trips=(trips_col, "sum"),
            total_replaced_time_minutes=(time_col, "sum"),
            total_residents=(residents_col, "sum"),
        )
    )
    result = {
        "data_scope": (
            "Virginia Eastern Shore pharmacy delivery scenario in the public "
            "OBrien drone-delivery dataset; not China policy evidence."
        ),
        "total_rows_in_file": int(len(data)),
        "rows_included_in_analysis": int(len(included)),
        "share_serviceable_2027": float(
            included["drone_serviceable"].mean()
        ),
        "total_replaced_trips_2027": float(included[trips_col].sum()),
        "total_replaced_time_minutes_2027": float(included[time_col].sum()),
        "replaced_time_minutes_per_resident_2027": float(
            included[time_col].sum() / included[residents_col].sum()
        ),
        "serviceability_groups": grouped.to_dict(orient="records"),
        "interpretation": (
            "The exercise validates that a real access surface can generate large "
            "task-level time savings in selected locations. It does not identify "
            "land capitalization or firm entry and should not be extrapolated to China."
        ),
    }
    (OUTPUT / "external_drone_access.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    grouped.to_csv(OUTPUT / "external_drone_access.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
