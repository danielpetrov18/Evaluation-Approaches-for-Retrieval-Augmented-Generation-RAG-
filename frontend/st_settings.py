"""Showcases settings and system information."""

from datetime import datetime
from r2r import R2RException
import streamlit as st
from streamlit.errors import Error
# No error, just annoying. Pylint complains about the import.
#             |
#             v
from st_app import load_client # pylint: disable=E0401

def format_uptime(seconds):
    """Convert seconds to a human-readable format."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{int(days)} days")
    if hours > 0:
        parts.append(f"{int(hours)} hours")
    if minutes > 0:
        parts.append(f"{int(minutes)} minutes")
    if seconds > 0 or not parts:
        parts.append(f"{int(seconds)} seconds")

    return ", ".join(parts)

def get_current_time():
    """Return current time formatted nicely."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def check_health():
    """Check health"""
    try:
        message = load_client().system.health()
        st.success(f"Service Status: {message.results.message}", icon="✅")
    except R2RException as r2re:
        st.error(f"Error checking health: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def check_status():
    """Check status"""
    try:
        with st.spinner(text="Fetching system status..."):
            status = load_client().system.status().results

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="CPU Usage", value=f"{status.cpu_usage}%")
            st.metric(label="Memory Usage", value=f"{status.memory_usage}%")

        with col2:
            st.markdown("### Time Information")
            st.markdown(f"**Start Time:** {status.start_time}")

            # Format uptime in a more readable way
            uptime_formatted = format_uptime(status.uptime_seconds)
            st.markdown(f"**Uptime:** {uptime_formatted}")

        st.caption(f"Last updated: {get_current_time()}")
    except R2RException as r2re:
        st.error(f"Error fetching status: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def check_settings():
    """Check settings"""
    try:
        settings = load_client().system.settings().results.config
        if settings and isinstance(settings, dict):
            st.write("**R2R Backend Settings:**")

            for key, value in settings.items():
                with st.expander(key, expanded=False):
                    if isinstance(value, dict):
                        st.json(value)
                    else:
                        st.write(value)
        else:
            st.info("No backend settings found.")
    except R2RException as r2re:
        st.error(f"Error fetching settings: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

if __name__ == "__page__":
    st.title("⚙️ Settings & System Information")

    tab_health, tab_status, tab_settings = st.tabs(
        [
            "Health", 
            "Status", 
            "Settings"
        ]
    )

    with tab_health:
        st.markdown("**Health Check**")

        if st.button(label="Check health"):
            check_health()

    with tab_status:
        st.markdown("### System Status")

        if st.button(label="Refresh Status"):
            check_status()

    with tab_settings:
        st.markdown("**Configuration Settings**")
        st.markdown("* Settings are read-only from here.")
        st.markdown("* To change them, update the server configuration and environment variables.")

        if st.button(label="Check configs"):
            check_settings()
