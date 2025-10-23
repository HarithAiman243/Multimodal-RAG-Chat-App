import streamlitUi as st
import yaml
import uuid
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.openai_chain import RAGChain
from src.utils import save_chat_history, load_chat_history, get_saved_sessions

from src.app_config import (
    CAMPAIGN_OBJECTIVES,
    TARGET_MARKETS,
    AGE_GROUPS,
    INDUSTRIES,
    AD_FORMATS
)

# --- Page Configuration ---
st.set_page_config(page_title="Meta Ads RAG Chatbot", layout="wide")

def check_password():
    """Returns `True` if the user enters the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect.")
    return False

def run_app():
    """
    The main function that runs the chatbot application after successful authentication.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # --- Sidebar for Conversation Management & Filters ---
    with st.sidebar:
        st.title("Conversation History")
        
        if st.button("➕ New Chat"):
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        # --- 📊 Strategic Filters ---
        st.markdown("---")
        st.markdown("## Analytical Filters")

        campaign_objective = st.selectbox(
            "Campaign Objective:",
            options=CAMPAIGN_OBJECTIVES
        )
        target_market = st.selectbox(
            "Target Market:",
            options=TARGET_MARKETS
        )
        age_groups = st.selectbox(
            "Age Group:",
            options=AGE_GROUPS
        )
        industry = st.selectbox(
            "Industry:",
            options=INDUSTRIES
        )
        ad_format = st.selectbox(
            "Ad Format:",
            options=AD_FORMATS
        )

        # --- 🌐 Language Selection ---
        st.markdown("---")
        st.markdown("## Language Settings")
        output_language = st.selectbox(
            "Select response language:",
            ("English", "Malay", "Mandarin")
        )
        
        # --- Past Conversations ---
        st.markdown("---")
        saved_sessions = get_saved_sessions()
        if not saved_sessions:
            st.write("No saved conversations yet.")
        else:
            st.markdown("Past Conversations:")
            for session in saved_sessions:
                session_id = session['session_id']
                title = f"Chat from {session['last_modified'].strftime('%Y-%m-%d %H:%M')}"
                if st.button(title, key=session_id, use_container_width=True):
                    st.session_state.session_id = session_id
                    st.rerun()

    # --- Main Chat Interface ---
    st.title("🤖 Multimodal RAG LLM Chatbot")
    st.caption("An AI assistant for the Inhouse Digital Marketing Team")

    history_key = f"history_{st.session_state.session_id}"
    chat_history = StreamlitChatMessageHistory(key=history_key)

    if not chat_history.messages:
        chat_history.messages = load_chat_history(st.session_state.session_id)

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_input := st.chat_input("Ask about your active ad campaigns..."):
        st.chat_message("human").write(user_input)
        
        with st.spinner("Analyzing data and generating response..."):
            # --- Construct the full prompt with context from filters ---
            context_parts = []
            if campaign_objective != "All":
                context_parts.append(f"for the '{campaign_objective}' objective")
            if target_market != "All":
                context_parts.append(f"targeting the '{target_market}' market")
            if age_groups != "All":
                context_parts.append(f"focusing on the '{age_groups}' age group")
            if industry != "All":
                context_parts.append(f"in the '{industry}' industry")
            if ad_format != "All":
                context_parts.append(f"using the '{ad_format}' format")

            context_preamble = ""
            if context_parts:
                context_preamble = "Analyze the following query " + " ".join(context_parts) + ". "

            final_input = f"{context_preamble}Answer in {output_language}. User question: {user_input}"
            # --------------------------------------------------------
            
            rag_chain = RAGChain(chat_history)
            response = rag_chain.run(final_input)
            
            st.chat_message("ai").write(response)
            
            save_chat_history(st.session_state.session_id, chat_history.messages)

# --- Application Gatekeeper ---
if check_password():
    run_app()