"""Showcases settings and system information."""

import sys
from pathlib import Path
from datetime import datetime
from r2r import R2RException
import streamlit as st
from streamlit.errors import Error
from st_app import load_client

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from settings import Settings

# Helper functions for formatting
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

@st.cache_resource
def get_settings_handler():
    """Get the settings handler."""
    return Settings(client=load_client())

if __name__ == "__page__":
    st.title("⚙️ Settings & System Information")

    tab_health, tab_status, tab_settings = st.tabs(["Health", "Status", "Settings"])

    with tab_health:
        st.markdown("**Health Check**")
        health_btn = st.button("Check health")
        if health_btn:
            try:
                message = get_settings_handler().health()
                st.success(f"Service Status: {message.results.message}", icon="✅")
            except R2RException as r2re:
                st.error(f"Error checking health: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with tab_status:
        st.markdown("### System Status")
        status_btn = st.button("Refresh Status")

        if status_btn:
            try:
                with st.spinner("Fetching system status..."):
                    status = get_settings_handler().status().results

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="CPU Usage", value=f"{status.cpu_usage}%")
                    st.metric(label="Memory Usage", value=f"{status.memory_usage}%")

                with col2:
                    st.markdown("### Time Information")
                    st.markdown(f"**Start Time:** {status.start_time}")

                    # Format uptime in a more readable way
                    UPTIME_FORMATTED = format_uptime(status.uptime_seconds)
                    st.markdown(f"**Uptime:** {UPTIME_FORMATTED}")

                st.caption(f"Last updated: {get_current_time()}")
            except R2RException as r2re:
                st.error(f"Error fetching status: {str(r2re)}")
            except Error as e:
                st.error(f"Unexpected error: {str(e)}")

    with tab_settings:
        st.markdown("**Configuration Settings**")
        st.markdown("* Settings are read-only from here.")
        st.markdown("* To change them, update the server configuration and environment variables.")

        config_btn = st.button("Check configs")
        if config_btn:
            try:
                settings = get_settings_handler().settings().results.config
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
