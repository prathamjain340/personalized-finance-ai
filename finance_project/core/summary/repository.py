# Domain summaries
# app/core/summary/repository.py

from typing import Optional


def get_domain_summary(user_id: str, domain: str) -> Optional[str]:
    """
    Fetches the derived summary for a given user and domain.

    This summary is:
    - Generated asynchronously
    - Read-only during request handling
    - Used to stabilize tone and personalization
    """

    if domain != "finance":
        return None

    # TEMPORARY STUB SUMMARY
    # Replace with DB-backed implementation later
    return (
        # "The user tends to be cautious with money, prefers understanding risks "
        # "before acting, and benefits from structured, step-by-step explanations "
        # "when making financial decisions."
    )
