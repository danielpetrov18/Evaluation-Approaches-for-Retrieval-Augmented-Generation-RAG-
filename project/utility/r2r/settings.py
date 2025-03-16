"""Simple file for separation the GUI from the logic as much as possible."""

import datetime
import streamlit as st
from streamlit.errors import Error
from r2r import R2RException, R2RClient

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
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def check_health(client: R2RClient):
    """Check health"""
    try:
        message = client.system.health()
        st.success(f"Service status: {message.results.message}")
    except R2RException as r2re:
        st.error(f"Error checking health: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def check_status(client: R2RClient):
    """Check status"""
    try:
        with st.spinner(text="Fetching system status..."):
            status = client.system.status().results

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="CPU Usage", value=f"{status.cpu_usage}%")
            st.metric(label="Memory Usage", value=f"{status.memory_usage}%")

        with col2:
            st.markdown("### Time Information")
            st.markdown(f"**Start Time:** {status.start_time}")

            uptime_formatted = format_uptime(status.uptime_seconds)
            st.markdown(f"**Uptime:** {uptime_formatted}")

        st.caption(f"Last updated: {get_current_time()}")
    except R2RException as r2re:
        st.error(f"Error fetching status: {str(r2re)}")
    except Error as e:
        st.error(f"Unexpected error: {str(e)}")

def check_settings(client: R2RClient):
    """Check settings"""
    try:
        settings = client.system.settings().results.config
        if settings:
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
