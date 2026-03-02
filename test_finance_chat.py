# test_finance_chat.py

from finance_project.domains.finance.engine import FinanceEngine
from finance_project.core.chat.session import ChatSession
from finance_project.core.chat.followup import detect_followup_intent


def run_chat():
    engine = FinanceEngine()
    session = ChatSession(user_id="user_1", domain="finance")

    print("Finance Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat.")
            break

        # Build contextual query using last turn
        contextual_query = session.build_query(user_input)

        # Detect followup intent
        followup_intent = detect_followup_intent(user_input)

        contextual_query = session.build_query(user_input)
        contextual_query = (
            f"FOLLOW-UP INTENT: {followup_intent}\n\n"
            f"{contextual_query}"
        )

        # Call finance engine
        response, memories_used = engine.handle_request(
            user_id="user_1",
            raw_query=contextual_query,
            session_memory_usage=session.session_memory_usage,
            conversation_stage=session.stage
        )

        print("\nAssistant:", response, "\n")

        # Update session with latest turn
        session.update_stage(followup_intent)
        session.update(
            user_message=user_input,
            assistant_response=response,
            memories_used=memories_used
        )

if __name__ == "__main__":
    run_chat()