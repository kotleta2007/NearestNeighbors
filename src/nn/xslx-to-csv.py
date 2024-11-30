import pandas as pd
from pathlib import Path

project_root = Path().absolute()
excel_dir = project_root / "files"

output_dir = project_root / "files" / "csv"
output_dir.mkdir(parents=True, exist_ok=True)

for excel_file in excel_dir.glob("*.xlsx"):
    all_sheets = pd.read_excel(excel_file, sheet_name=None)

    base_name = excel_file.stem.replace(" ", "-")
    for sheet_name, df in all_sheets.items():
        sheet_name = sheet_name.replace(" ", "-")
        csv_file = output_dir / f"{base_name}_{sheet_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}")

    sheet_names = list(all_sheets.keys())
