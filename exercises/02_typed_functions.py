
# Exercice 1: type hintsin using dict
def parse_user(raw: dict) -> dict[str, str | int]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected dict, got {type(raw).__name__}")
    try:
        name = raw["name"]
        age_raw = raw["age"]
    except KeyError as e:
        raise ValueError(f"missing required field {e.args[0]}") from e
    
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    
    try:
        age = int(age_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"age mist be coercible to int, got {age_raw!r}") from e
    if not 0 <= age <= 130:
        raise ValueError(f"age out of range [0, 130]:{age}")
    
    return {"name": name.strip(), "age": age}

# Exercise 2: list comprehension
def filter_adults(users: list[dict]) -> list[dict]:
    return [user for user in users if user["age"] >= 18]

# Exercise 3: summarize
def summarize(users: list[dict]) -> dict[str, int]:
    total = len(users)
    adults = len(filter_adults(users))
    average_age = sum(user["age"] for user in users) // total if total > 0 else 0
    return {"total": total, "adults": adults, "average_age": average_age}

if __name__ == "__main__":
    raw_users = [
        {"name": "Ana", "age": "30"},
        {"name": "Luis", "age": "17"},
        {"name": "  María  ", "age": "25"},
        {"name": "Pedro", "age": "15"},
        {"name": "Sofía", "age": "42"},
    ]

    users = [parse_user(raw) for raw in raw_users]
    print("Parsed users:", users)
    print("Adults:", filter_adults(users))
    print("Summary:", summarize(users))
