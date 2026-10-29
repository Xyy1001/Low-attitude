from __future__ import annotations

import json

from research_lib import PROCESSED, build_panel


def main() -> None:
    panel, quality = build_panel()
    summary = (
        f"Panel rows: {len(panel):,}\n"
        f"Cities: {panel['citycode'].nunique()}\n"
        f"Years: {int(panel['year'].min())}-{int(panel['year'].max())}\n"
        f"Main treated cities: {quality['main_treated_cities']}\n"
        f"Output: {PROCESSED / 'panel_city_year.csv'}\n"
    )
    print(summary)
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
