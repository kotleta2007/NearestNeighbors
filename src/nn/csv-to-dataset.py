import polars as pl
from pathlib import Path
from utils.french_mapping import convert_french_date

COUNTRIES = "BRISTOR_Zegoland", "INNOVIX_Elbonie", "INNOVIX_Floresland"


def process_country(country: str):
    in_path = Path() / "files" / "csv"
    out_path = Path() / "files" / "datasets"
    out_path.mkdir(parents=True, exist_ok=True)

    df_activity = (
        pl.read_csv(f"files/csv/{country}_Activity.csv")
        .pivot(
            on=["Data type", "Channel", "Product"],
            index=["Date", "Country"],
            values="Value",
        )
        .fill_null(0)
        .with_columns(pl.col("Date").map_elements(convert_french_date))
    )
    df_demand_volumes = (
        pl.read_csv(f"files/csv/{country}_Demand-volumes.csv")
        .pivot(
            on=["Data type", "Indication", "Product"],
            index=["Country", "Date"],
            values="Value",
        )
        .with_columns(pl.col("Date").map_elements(convert_french_date))
    )
    df_factory_volumes = pl.read_csv(
        f"files/csv/{country}_Ex-factory-volumes.csv"
    ).pivot(on=["Data type", "Product"], index=["Date", "Country"], values="Value")
    df_patients = (
        pl.read_csv(f"files/csv/{country}_Patient-numbers-and-share.csv")
        .pivot(
            on=["Data type", "Product", "Indication", "Measure"],
            index=["Country", "Date"],
            values="Value",
        )
        .with_columns(pl.col("Date").map_elements(convert_french_date))
    )
    df_voice = (
        pl.read_csv(f"files/csv/{country}_Share-of-Voice.csv")
        .pivot(
            on=["Data type", "Indication", "Products"],
            index=["Country", "Date"],
            values="Value",
        )
        .with_columns(pl.col("Date").map_elements(convert_french_date))
    )
    join_on = ["Country", "Date"]
    df = df_activity
    dfs = df_demand_volumes, df_factory_volumes, df_patients, df_voice
    for d in dfs:
        df = df.join(d, join_on)
    df.write_csv(out_path / f"{country}.csv")


if __name__ == "__main__":
    for c in COUNTRIES:
        process_country(c)
