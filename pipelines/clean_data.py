from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent.resolve()

def clean_data() -> None:
    dfs = extract()
    load(transform(dfs))

def extract() -> dict[str, pd.DataFrame]:
    selected_dir = ROOT / "data" / "selected"

    extracts: list[tuple[str, str]] = [
            ("customers", "customers_dataset.csv"),
            ("geolocations", "geolocations_dataset.csv"),
            ("items", "order_items_dataset.csv"),
            ("orders", "orders_dataset.csv"),
            ("products", "products_dataset.csv"),
            ("sellers", "sellers_dataset.csv")
        ]

    column_types = {
            "customers": {"customer_zip_code_prefix": "str"},
            "sellers": {"seller_zip_code_prefix": "str"},
            "geolocations": {"geolocation_zip_code_prefix": "str"}
        }

    return {
            df_name: pd.read_csv(
                selected_dir / file_name,
                dtype=column_types.get(df_name)
            ) for df_name, file_name in extracts
        }

def clean_geol(df: pd.DataFrame) -> pd.DataFrame:
    df_geol = df.copy()
    df_geol = df_geol.drop_duplicates(
        subset=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng"
        ]
    )

    df_geol = df_geol.groupby(
        by="geolocation_zip_code_prefix",
        as_index=False,

    )[["geolocation_lat", "geolocation_lng"]].median()

    return df_geol

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df_orders = df.copy()

    date_columns = [
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        df_orders[column] = pd.to_datetime(df_orders[column])

    approved = df_orders["order_approved_at"]
    delivered = df_orders["order_delivered_customer_date"]
    estimated = df_orders["order_estimated_delivery_date"]

    delivered_before_approval = delivered < approved
    estimate_before_approval_day = (
        estimated.dt.normalize() < approved.dt.normalize()
    )

    df_orders = df_orders.loc[
        ~(delivered_before_approval | estimate_before_approval_day)
    ]

    df_orders = df_orders.loc[df_orders["order_status"] == "delivered"].dropna(
        subset=["order_id", "customer_id", *date_columns]
    )

    return df_orders

def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df_products = df.copy()
    df_products["product_weight_g"] = df_products["product_weight_g"].replace(
        0, float("nan")
    )

    return df_products

def seller_distance_km(items: pd.DataFrame) -> pd.Series:
    customer_lat = np.radians(items["customer_lat"])
    seller_lat = np.radians(items["seller_lat"])
    latitude_difference = customer_lat - seller_lat
    longitude_difference = np.radians(items["customer_lng"] - items["seller_lng"])

    # Haversine distance between the representative ZIP prefix coordinates.
    haversine = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(customer_lat) * np.cos(seller_lat)
        * np.sin(longitude_difference / 2) ** 2
    )
    earth_radius_km = 6371.0
    distance = 2 * earth_radius_km * np.arcsin(
        np.sqrt(haversine.clip(0, 1))
    )
    return pd.Series(distance, index=items.index, name="seller_distance_km")


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    by_order = items.groupby("order_id")
    summary = by_order.agg(
        item_count=("order_item_id", "size"),
        distinct_product_count=("product_id", "nunique"),
        distinct_seller_count=("seller_id", "nunique"),
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("item_volume_cm3", "sum"),
        max_seller_distance_km=("seller_distance_km", "max"),
    )

    measurement_columns = {
        "product_weight_g": "total_weight_g",
        "item_volume_cm3": "total_volume_cm3",
        "seller_distance_km": "max_seller_distance_km",
    }
    known_counts = by_order[list(measurement_columns)].count()
    for item_column, order_column in measurement_columns.items():
        # These summaries require a measurement for every item in the order.
        complete = known_counts[item_column].eq(summary["item_count"])
        summary[order_column] = summary[order_column].where(complete)

    return summary.reset_index()


def transform(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df_orders = clean_orders(dfs["orders"])
    df_geol = clean_geol(dfs["geolocations"])
    df_products = clean_products(dfs["products"])

    customer_coordinates = df_geol.rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix",
        "geolocation_lat": "customer_lat",
        "geolocation_lng": "customer_lng",
    })
    customers = dfs["customers"].merge(
        customer_coordinates, on="customer_zip_code_prefix", how="left",
        validate="many_to_one",
    )
    orders = df_orders.merge(
        customers, on="customer_id", how="left", validate="many_to_one",
    )

    seller_coordinates = df_geol.rename(columns={
        "geolocation_zip_code_prefix": "seller_zip_code_prefix",
        "geolocation_lat": "seller_lat",
        "geolocation_lng": "seller_lng",
    })
    sellers = dfs["sellers"][["seller_id", "seller_zip_code_prefix"]].merge(
        seller_coordinates, on="seller_zip_code_prefix", how="left",
        validate="many_to_one",
    )

    dimensions = ["product_length_cm", "product_height_cm", "product_width_cm"]
    product_columns = ["product_id", "product_weight_g", *dimensions]
    items = dfs["items"].merge(
        orders[["order_id", "customer_lat", "customer_lng"]], on="order_id",
        how="inner", validate="many_to_one",
    ).merge(
        sellers[["seller_id", "seller_lat", "seller_lng"]], on="seller_id",
        how="left", validate="many_to_one",
    ).merge(
        df_products[product_columns], on="product_id", how="left",
        validate="many_to_one",
    )
    items["item_volume_cm3"] = items[dimensions].prod(axis=1, skipna=False)
    items["seller_distance_km"] = seller_distance_km(items)

    approved_day = orders["order_approved_at"].dt.normalize()
    estimated_day = orders["order_estimated_delivery_date"].dt.normalize()
    delivered_day = orders["order_delivered_customer_date"].dt.normalize()
    orders["days_from_approval_to_estimate"] = (estimated_day - approved_day).dt.days
    orders["is_late"] = delivered_day.gt(estimated_day).astype(int)

    # Keep the order ID and dates for splitting, label availability, and auditing.
    order_columns = [
        "order_id", "order_approved_at", "order_estimated_delivery_date",
        "order_delivered_customer_date", "customer_city", "customer_state",
        "days_from_approval_to_estimate", "is_late",
    ]
    return orders[order_columns].merge(
        aggregate_items(items), on="order_id", how="left", validate="one_to_one",
    )

def load(df: pd.DataFrame) -> None:
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)

    df.to_csv(
        processed_dir / "logistic_abt.csv",
        index=False
    )


if __name__ == "__main__":
    clean_data()
