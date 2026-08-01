import json
import os
import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from .retriever import load_vector_store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT,
    "data",
    "processed",
    "simple_house_properties_clean.csv",
)

MODEL_PATH = os.path.join(ROOT, "models", "best_regression_model.joblib")
properties = pd.read_csv(DATA_PATH)
AVAILABLE_CITIES = set(properties["city"].dropna())

def price_segment(price):
    if price < 50:
        return "Affordable"
    if price <= 150:
        return "Mid-range"
    return "Premium"

def load_model():
    return joblib.load(MODEL_PATH)

def clean_text(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()

    if value.casefold() in {"", "nan", "none", "unknown"}:
        return None
    return value

def parse_query(question):
    """Convert a property question into city, BHK, and budget filters."""
    text = question.casefold()
    # casefold makes formats such as Pune, PUNE, and pune behave the same.

    # handles 80lakhs, 2cr
    for unit in ["lakhs", "lakh", "lacs", "lac", "crores", "crore", "cr"]:
        text = text.replace(unit, f" {unit}")
        
    # handles 80L
    raw_tokens = text.replace(",", " ").split()
    tokens = []
    for token in raw_tokens:
        if token.endswith("l"):
            try:
                float(token[:-1])
            except ValueError:
                tokens.append(token)
            else:
                tokens.extend([token[:-1], "l"])
        else:
            tokens.append(token)

    out = {}
    for city in properties["city"].dropna().unique():
        if city.casefold() in text:
            out["city"] = city
            break

    if "delhi" in text and "city" not in out:
        out["city"] = "New Delhi"

    if "bengaluru" in text and "city" not in out:
        out["city"] = "Bangalore"

    if "gurugram" in text and "city" not in out:
        out["city"] = "Gurgaon"

    # Searches for compact or hyphenated tokens such as 3bhk, 3BHK, and 3-bhk.
    for token in tokens:
        compact_token = token.replace("-", "")
        if compact_token.endswith("bhk") and compact_token[:-3].isdigit():
            out["bhk"] = float(compact_token[:-3])
            break

    # Searches for separated BHK tokens such as "2 bhk" or "2 BHK".
    for i, token in enumerate(tokens[:-1]):
        if token.isdigit() and tokens[i + 1] == "bhk":
            out["bhk"] = float(token)
            break

    budget = None
    crore = None
    for i, token in enumerate(tokens[:-1]):
        # Budget examples: under 80 lakh, below 1 crore, or budget 75.
        if token in {
            "under",
            "below",
            "budget",
            "upto",
            "within",
            "in",
            "up",
            "up-to",
            "up to"
        }:
            nxt = tokens[i + 1]
            try:
                value = float(nxt)
            except ValueError:
                continue

            if i + 2 < len(tokens) and tokens[i + 2] in {"crore", "crores", "cr"}:
                # Prices are stored in lakh, so convert crore to lakh.
                crore = value

            elif i + 2 < len(tokens) and tokens[i + 2] in {"lakh", "lakhs", "lac", "lacs", "l"}:
                # Lakh variants are already in the dataset's price unit.
                budget = value

            else:
                # If no unit is supplied, treat the number as lakh.
                budget = value
            break

    if crore is not None:
        out["max_price_lakh"] = crore * 100

    elif budget is not None:
        out["max_price_lakh"] = budget

    return out

def city_error(filters):
    if "city" not in filters:
        return "Please mention a city available in the dataset."

    if filters["city"] in AVAILABLE_CITIES:
        return None

    return f"Data unavailable for {filters['city']}."

# sets default values in case of unavailable data
def typical(rows, col, default):
    if col not in rows.columns:
        return default

    vals = rows[col].dropna()
    if vals.empty:
        return default

    try:
        return float(vals.astype(float).median())
    except (TypeError, ValueError):
        pass

    return vals.mode().iloc[0] if not vals.mode().empty else default

def encode_for_model(df, model):
    categorical_columns = [
        "city",
        "society",
        "status",
        "property_type",
        "transaction",
        "furnishing",
        "ownership",
    ]

    # one hot encoding
    x = pd.get_dummies(df, columns=categorical_columns)
    x = x.reindex(columns=model.feature_names_in_, fill_value=0)
    return x

def estimate_typical_price(filters):
    err = city_error(filters)
    if err:
        return err

    if "bhk" not in filters:
        return "Please mention BHK, for example: 2 BHK in Nagpur."

    rows = properties[(properties["city"] == filters["city"]) & (properties["bhk"] == filters["bhk"])]
    if rows.empty:
        return f"No {filters['bhk']:g} BHK records are available for {filters['city']}."

    sample = pd.DataFrame([{
        "area_sqft": typical(rows, "area_sqft", 1000.0),
        "bhk": filters["bhk"],
        "bathrooms": typical(rows, "bathrooms", filters["bhk"]),
        "balconies": typical(rows, "balconies", 1.0),
        "parking_count": typical(rows, "parking_count", 0.0),
        "current_floor": typical(rows, "current_floor", 2.0),
        "total_floors": typical(rows, "total_floors", 4.0),
        "city": filters["city"],
        "society": typical(rows, "society", "Unknown"),
        "status": typical(rows, "status", "Ready to Move"),
        "property_type": typical(rows, "property_type", "Flat"),
        "transaction": typical(rows, "transaction", "Resale"),
        "furnishing": typical(rows, "furnishing", "Semi-Furnished"),
        "ownership": typical(rows, "ownership", "Unknown"),
    }])

    model = load_model()
    pred = model.predict(encode_for_model(sample, model))[0]
    price = round(float(np.expm1(pred)), 2)

    return {
        "city": filters["city"],
        "bhk": filters["bhk"],
        "estimated_price_lakh": price,
        "price_segment": price_segment(price),
        "typical_area_sqft": round(float(sample.loc[0, "area_sqft"]), 2),
        "similar_records_used": int(len(rows)),
        "dataset_median_asking_price_lakh": round(float(rows["price_lakh"].median()), 2),
    }

def find_deals(filters):
    err = city_error(filters)
    if err:
        return err

    if "bhk" not in filters or "max_price_lakh" not in filters:
        return "Please mention BHK and budget, for example: 2 BHK under 80 lakh in Pune."

    rows = properties[
        (properties["city"] == filters["city"])
        & (properties["bhk"] == filters["bhk"])
    ].dropna(subset=["price_lakh", "area_sqft", "bhk"])

    if rows.empty:
        return "No comparable properties found for this city and BHK."

    user = float(filters["max_price_lakh"])
    budget_rows = rows[rows["price_lakh"] <= user]
    typical_est = estimate_typical_price(filters)
    fair = (
        typical_est["estimated_price_lakh"]
        if isinstance(typical_est, dict)
        else None
    )

    return {
        "city": filters["city"],
        "bhk": filters["bhk"],
        "user_price_lakh": round(user, 2),
        "comparable_records": int(len(rows)),
        "average_price_lakh": round(float(rows["price_lakh"].mean()), 2),
        "median_price_lakh": round(float(rows["price_lakh"].median()), 2),
        "price_vs_average_lakh": round(user - float(rows["price_lakh"].mean()), 2),
        "price_vs_median_lakh": round(user - float(rows["price_lakh"].median()), 2),
        "properties_within_budget": int(len(budget_rows)),
        "typical_area_within_budget_sqft": round(float((budget_rows if not budget_rows.empty else rows)["area_sqft"].median()), 2),
        "expected_fair_price_lakh": fair,
        "instruction": "Give a brief summary comparing user_price_lakh with average_price_lakh, median_price_lakh, and expected_fair_price_lakh. Then give buy or bargain advice.",
    }

def search_rag_context(question, filters):
    err = city_error(filters)
    if err:
        return err

    try:
        docs = load_vector_store().similarity_search(question, k=30)

    except Exception:
        docs = []
    
    out = []
    for doc in docs:
        md = doc.metadata
        if md.get("city") != filters["city"]:
            continue

        if "bhk" in filters and md.get("bhk") != filters["bhk"]:
            continue

        if ("max_price_lakh" in filters and md.get("price_lakh", 0) > filters["max_price_lakh"]):
            continue

        sid = md.get("source_id")
        row = None
        if sid and str(sid).startswith("property_"):
            tail = str(sid).split("_", 1)[1]

            if tail.isdigit():
                idx = int(tail)

                if idx < len(properties):
                    row = properties.iloc[idx]

        out.append({
            "source_id": sid,
            "city": md.get("city"),
            "location": clean_text((row.get("society") if row is not None else None) or (row.get("title") if row is not None else None) or md.get("city")),
            "description": clean_text(row.get("description")) if row is not None else None,
            "bhk": md.get("bhk"),
            "price_lakh": md.get("price_lakh"),
            "area_sqft": float(row.get("area_sqft")) if row is not None and pd.notna(row.get("area_sqft")) else None,
            "furnishing": clean_text(row.get("furnishing")) if row is not None else None,
            "bathrooms": float(row.get("bathrooms")) if row is not None and pd.notna(row.get("bathrooms")) else None,
            "balconies": float(row.get("balconies")) if row is not None and pd.notna(row.get("balconies")) else None,
            "parking_count": float(row.get("parking_count")) if row is not None and pd.notna(row.get("parking_count")) else None,
            "property_type": clean_text(row.get("property_type")) if row is not None else None,
            "status": clean_text(row.get("status")) if row is not None else None,
            "transaction": clean_text(row.get("transaction")) if row is not None else None,
            "amount_below_budget_lakh": round(float(filters["max_price_lakh"]) - float(md.get("price_lakh", 0)), 2) if "max_price_lakh" in filters else None,
            "listing_text": doc.page_content[:800],
        })

    return out[:3] if out else []

def to_json(key, filters, result):
    return json.dumps({"parsed_inputs": filters, key: result})

@tool
def estimate_typical_price_tool(question: str) -> str:
    """Estimate a typical property price for a city and BHK."""
    filters = parse_query(question)
    return to_json(
        "regression_model_output",
        filters,
        estimate_typical_price(filters),
    )

@tool
def find_deals_tool(question: str) -> str:
    """Check whether a price looks like a good deal for the given city and BHK."""
    filters = parse_query(question)
    return to_json(
        "regression_model_output",
        filters,
        find_deals(filters),
    )

@tool
def search_rag_context_tool(question: str) -> str:
    """Find up to three matching property listings with factual details."""
    filters = parse_query(question)
    return to_json(
        "chroma_retrieval_output",
        filters,
        search_rag_context(question, filters),
    )

@tool
def city_market_summary_tool(city: str) -> str:
    """Return a simple city market summary from the dataset."""
    error = city_error({"city": city})
    if error:
        return error

    df = properties[properties["city"] == city]
    return json.dumps({
        "city": city,
        "listings": int(len(df)),
        "median_price_lakh": round(float(df["price_lakh"].median()), 2),
        "average_price_lakh": round(float(df["price_lakh"].mean()), 2),
        "median_area_sqft": round(float(df["area_sqft"].median()), 2),
        "most_common_bhk": float(df["bhk"].mode().iloc[0]),
    })

HOUSE_TOOLS = [
    estimate_typical_price_tool,
    find_deals_tool,
    search_rag_context_tool,
    city_market_summary_tool,
]
