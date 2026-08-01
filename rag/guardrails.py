PROPERTY_HINTS = {
    "house",
    "home",
    "flat",
    "apartment",
    "property",
    "plot",
    "villa",
    "room",
    "buy",
    "buying",
    "sell",
    "selling",
    "price",
    "budget",
    "deal",
    "city",
    "bhk",
    "bedroom",
    "bathroom",
    "furnishing",
    "ownership",
    "area",
    "sqft",
    "society",
    "market",
    "locality",
    "location",
    "pune",
    "mumbai",
    "bangalore",
    "delhi",
    "noida",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "chennai",
    "kolkata",
    "jaipur",
    "surat",
    "thane",
    "nagpur",
    "lucknow",
    "varanasi",
    "chandigarh",
}

BLOCKED_HINTS = {
    "medical",
    "medicine",
    "legal",
    "lawsuit",
    "court",
    "divorce",
    "loan approval",
    "investment advice",
    "stock",
    "crypto",
    "bank account",
    "password",
    "hacking",
}

def is_property_related(text: str) -> bool:
    text = (text or "").casefold()
    return any(hint in text for hint in PROPERTY_HINTS)

def should_block_query(text: str) -> bool:
    text = (text or "").casefold()
    return (
        any(hint in text for hint in BLOCKED_HINTS)
        or not is_property_related(text)
    )

def refusal_message() -> str:
    return (
        "I can only help with property-related questions like house prices, budgets, "
        "cities, locations, features, or market summaries."
    )