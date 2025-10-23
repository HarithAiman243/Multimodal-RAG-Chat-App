import streamlit as st
from src.app_config import (
    CAMPAIGN_OBJECTIVES,
    TARGET_MARKETS,
    INDUSTRIES,
    AD_FORMATS
)

def render_sidebar(saved_sessions: list):
    """
    Renders the sidebar UI components and returns user actions.
    
    Args:
        saved_sessions (list): A list of saved session dictionaries.
    
    Returns:
        str or None: The session_id of a selected chat, or None if no selection is made.
    """
    st.title("Chat Conversation")
        
    if st.button("➕ New Chat"):
        st.session_state.session_id = None # Signal to create a new session
        st.rerun()

    st.markdown("---")
    
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

def render_filters() -> dict:
    """
    Renders the collapsible filter section and returns the selected filter values.
    
    Returns:
        dict: A dictionary containing the current values of all filters.
    """
    st.markdown("## Set Analytical Filters")
    filters = {}
    filters['campaign_objective'] = st.selectbox("Campaign Objective:", options=CAMPAIGN_OBJECTIVES)
    filters['target_market'] = st.selectbox("Target Market:", options=TARGET_MARKETS)
    filters['industry'] = st.selectbox("Industry:", options=INDUSTRIES)
    filters['ad_format'] = st.selectbox("Ad Format:", options=AD_FORMATS)
    
    st.markdown("---")
    st.markdown("## Language Settings")
    filters['output_language'] = st.selectbox("Response Language:", ("English", "Malay", "Mandarin"))

    return filters

def render_chat_interface(chat_history):
    """
    Renders the main chat message display area.
    """
    st.title("🤖 Multimodal RAG LLM Chatbot")
    st.caption("An AI assistant/ideation for the Inhouse INVOKE's Digital Marketing Team")

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)