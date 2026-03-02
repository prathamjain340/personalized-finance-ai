from finance_project.domains.finance.engine import FinanceEngine


if __name__ == "__main__":
    engine = FinanceEngine()

    response = engine.handle_request(
        user_id="user_1",
        raw_query="Where should I invest my money?"
    )

    print("\n=== FINAL RESPONSE ===")
    print(response)

    response = engine.handle_request(
        user_id="user_2",
        raw_query="Where should I invest my money?"
    )

    print("\n=== FINAL RESPONSE ===")
    print(response)

    response = engine.handle_request(
        user_id="user_3",
        raw_query="Where should I invest my money?"
    )

    print("\n=== FINAL RESPONSE ===")
    print(response)