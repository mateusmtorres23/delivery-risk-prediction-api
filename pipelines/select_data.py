from pathlib import Path
import pandas as pd


ROOT = Path(__file__).parent.parent.resolve()

def select_data() -> None:
    dfs: dict[str, pd.DataFrame] = extract()

    transformed_dfs: dict[str, pd.DataFrame] = transform(dfs) 

    load(transformed_dfs)

def extract() -> dict[str, pd.DataFrame]:
    raw_dir = ROOT / "data" / "raw"

    extracts: list[tuple[str, str]] = [
        ("customers", "olist_customers_dataset.csv"),
        ("geolocations", "olist_geolocation_dataset.csv"),
        ("items", "olist_order_items_dataset.csv"),
        ("orders", "olist_orders_dataset.csv"),
        ("products", "olist_products_dataset.csv"),
        ("translations", "product_category_name_translation.csv"),
        ("sellers", "olist_sellers_dataset.csv")
    ]

    column_types = {
        "customers": {"customer_zip_code_prefix": "str"},
        "sellers": {"seller_zip_code_prefix": "str"},
        "geolocations": {"geolocation_zip_code_prefix": "str"}
    }

    return {
        df_name: pd.read_csv(
            raw_dir / file_name,
            dtype=column_types.get(df_name)
        ) for df_name, file_name in extracts
    }

def transform(
        dfs: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    transformed_dfs = dfs.copy()
    
    df_customers = dfs["customers"].drop(
            columns=["customer_unique_id"]
    )

    df_geol = dfs["geolocations"].drop(
        columns=["geolocation_city", "geolocation_state"]
    )

    df_items = dfs["items"].drop(columns=["shipping_limit_date"])

    df_orders = dfs["orders"].drop(
        columns=["order_delivered_carrier_date"]
    )

    df_products = dfs["products"].merge(
        dfs["translations"],
        on="product_category_name",
        how="left",
        validate="many_to_one"
    )

    df_products = df_products.drop(
        columns="product_category_name"
    ).rename(
        columns={"product_category_name_english": "product_category_name"}
    )
    df_products = df_products.drop(
        columns=[
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty"
        ]
    )

    transformed_dfs["customers"] = df_customers
    transformed_dfs["geolocations"] = df_geol
    transformed_dfs["items"] = df_items
    transformed_dfs["orders"] = df_orders
    transformed_dfs["products"] = df_products

    return transformed_dfs

def load(dfs: dict[str, pd.DataFrame]) -> None:
    select_dir = ROOT / "data" / "selected"
    select_dir.mkdir(exist_ok=True)

    exports: list[tuple[pd.DataFrame, str]] = [
        (dfs["customers"], "customers_dataset"),
        (dfs["geolocations"], "geolocations_dataset"),
        (dfs["items"], "order_items_dataset"),
        (dfs["orders"], "orders_dataset"),
        (dfs["products"], "products_dataset"),
        (dfs["sellers"], "sellers_dataset")
    ]

    for df, stem in exports:
        df.to_csv(
            select_dir / f"{stem}.csv",
            index=False
        )

if __name__ == "__main__":
    select_data()
