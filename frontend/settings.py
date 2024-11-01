# pages/2_⚙️_settings.py
import streamlit as st

# Custom CSS for settings page
st.markdown("""
    <style>
    .settings-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stMarkdown h3 {
        color: #4A4A4A;
    }
    </style>
""", unsafe_allow_html=True)

#st.title("⚙️ Chatbot Settings")

# Settings card for LLM parameters
#st.markdown('<div class="settings-card">', unsafe_allow_html=True)
#st.subheader("🤖 Language Model Parameters")

#st.markdown('</div>', unsafe_allow_html=True)

# Additional settings sections could be added here
#st.markdown('<div class="settings-card">', unsafe_allow_html=True)
#st.subheader("📊 Logging & Privacy")
#st.write("Coming soon: Options to manage conversation logging and privacy settings.")
#st.markdown('</div>', unsafe_allow_html=True)