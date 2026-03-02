# app/services/profile_client.py

import requests

PROFILE_BASE_URL = "https://api.eazyhaina.com/bsestar/api/v1/lhw/getfinancialprofile"


def fetch_financial_profile(user_id: str):
    try:
        response = requests.get(f"{PROFILE_BASE_URL}/{user_id}")
        data = response.json()

        if data.get("succeeded"):
            return data.get("data")

        return None

    except Exception:
        return None

def is_valid_profile(profile: dict) -> bool:
    if not profile:
        return False

    critical_fields = [
        "monthly_income",
        "monthly_expenses",
        "monthly_savings",
    ]

    for field in critical_fields:
        if profile.get(field, 0) <= 0:
            return False

    return True