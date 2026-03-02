# app/ui/streamlit_app.py

import streamlit as st
import streamlit.components.v1 as components

from finance_project.domains.finance.engine import FinanceEngine
from finance_project.core.chat.session import ChatSession
from finance_project.core.chat.followup import detect_followup_intent


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Finance Assistant", layout="wide")
st.title("Personal Finance Assistant")


# -------------------------------------------------
# SIDEBAR AVATAR (STABLE)
# -------------------------------------------------
with st.sidebar:
    st.markdown("### Virtual Advisor")
    st.video("app/assets/avatar.mp4", loop=True, autoplay=True, muted=True)


# -------------------------------------------------
# INIT SESSION STATE
# -------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = FinanceEngine()

if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession(
        user_id="user_1",
        domain="finance"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tts_text" not in st.session_state:
    st.session_state.tts_text = None


engine = st.session_state.engine
session = st.session_state.chat_session


# -------------------------------------------------
# RENDER CHAT HISTORY
# -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
# user_input = st.chat_input("Ask a finance question...")

# if user_input:

#     # 1️⃣ Show user message immediately
#     st.session_state.messages.append({
#         "role": "user",
#         "content": user_input
#     })

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):

#             # Build contextual query
#             contextual_query = session.build_query(user_input)
#             followup_intent = detect_followup_intent(user_input)

#             contextual_query = (
#                 f"FOLLOW-UP INTENT: {followup_intent}\n\n"
#                 f"{contextual_query}"
#             )

#             response, memories_used = engine.handle_request(
#                 user_id="user_1",
#                 raw_query=contextual_query,
#                 session_memory_usage=session.session_memory_usage,
#                 conversation_stage=session.stage
#             )

#             st.markdown(response)

#     # 2️⃣ Save assistant message
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": response
#     })

#     # 3️⃣ Store TTS text
#     st.session_state.tts_text = response

#     # 4️⃣ Update backend session
#     session.update(
#         user_message=user_input,
#         assistant_response=response,
#         memories_used=memories_used
#     )

user_input = st.chat_input("Ask a finance question...")

if user_input:

    # 1️⃣ Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
    # 2️⃣ Build contextual query
            contextual_query = session.build_query(user_input)
            followup_intent = detect_followup_intent(user_input)

            contextual_query = (
                f"FOLLOW-UP INTENT: {followup_intent}\n\n"
                f"{contextual_query}"
            )

            # 3️⃣ Generate response
            response, memories_used = engine.handle_request(
                user_id="user_1",
                raw_query=contextual_query,
                session_memory_usage=session.session_memory_usage,
                conversation_stage=session.stage
            )

    # 4️⃣ Append assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # 5️⃣ Store TTS text
    st.session_state.tts_text = response

    # 6️⃣ Update backend session
    session.update(
        user_message=user_input,
        assistant_response=response,
        memories_used=memories_used
    )

    # 7️⃣ Force immediate refresh
    st.rerun()



# -------------------------------------------------
# TEXT-TO-SPEECH (RUNS AFTER RENDER)
# -------------------------------------------------
if st.session_state.tts_text:

    components.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance({repr(st.session_state.tts_text)});
        msg.rate = 0.95;
        msg.pitch = 1.0;
        msg.lang = 'en-US';

        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0
    )

    # Clear after speaking
    st.session_state.tts_text = None
