import streamlit as st
import uuid
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.openai_chain import RAGChain
from src.utils import save_chat_history, load_chat_history, get_saved_sessions
from src import streamlitUi # <-- Import the new UI module

# --- Page Configuration ---
st.set_page_config(page_title="INVOKE's Multimodal RAG LLM Chatbot DM Speciality Tool", layout="wide")

def check_password():
    """Returns `True` if the user enters the correct password."""
    # ... (This function remains exactly the same as before) ...
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect.")
    return False

def run_app():
    """
    The main orchestrator for the application.
    """
    # --- Session State Initialization ---
    if "session_id" not in st.session_state or st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())

    # --- Render UI Components from ui.py ---
    with st.sidebar:
        saved_sessions = get_saved_sessions()
        streamlitUi.render_sidebar(saved_sessions)

        st.markdown("---")

        # Render filters section inside the sidebar
        filters = streamlitUi.render_filters()

    # Render the main chat display
    history_key = f"history_{st.session_state.session_id}"
    chat_history = StreamlitChatMessageHistory(key=history_key)
    if not chat_history.messages:
        chat_history.messages = load_chat_history(st.session_state.session_id)
    
    chat_container = st.container()
    with chat_container:
        streamlitUi.render_chat_interface(chat_history)
    
    # Render filters and get their current values
    #filters = streamlitUi.render_filters()

    # --- Handle Core Application Logic ---
    if user_input := st.chat_input("Ask about your active ad campaigns..."):
        # Display user message immediately
        st.chat_message("human").write(user_input)
        
        with st.spinner("Analyzing data and generating response..."):
            # Construct the prompt using filter values
            context_parts = []
            if filters['campaign_objective'] != "All":
                context_parts.append(f"for the '{filters['campaign_objective']}' objective")
            if filters['target_market'] != "All":
                context_parts.append(f"targeting the '{filters['target_market']}' market")
            if filters['industry'] != "All":
                context_parts.append(f"in the '{filters['industry']}' industry")
            if filters['ad_format'] != "All":
                context_parts.append(f"using the '{filters['ad_format']}' format")

            context_preamble = ""
            if context_parts:
                context_preamble = "Analyze the following query " + " ".join(context_parts) + ". "

            final_input = f"{context_preamble}Answer in {filters['output_language']}. User question: {user_input}"
            
            # Execute RAG chain
            rag_chain = RAGChain(chat_history)
            response = rag_chain.run(final_input)
            
            # Display AI response and save history
            st.chat_message("ai").write(response)
            save_chat_history(st.session_state.session_id, chat_history.messages)

# --- Application Gatekeeper ---
if check_password():
    run_app()