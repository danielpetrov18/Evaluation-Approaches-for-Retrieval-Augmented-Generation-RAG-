# pylint: disable=C0114
# pylint: disable=E0401 -> disable relative import error

import streamlit as st

from st_app import r2r_client
from backend.settings import (
    check_health,
    check_status,
    check_settings
)

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

        if st.button(label="Check health", key="health_check_button"):
            check_health(r2r_client())

    with tab_status:
        st.markdown("**System Status**")

        if st.button(label="Refresh Status", key="status_check_button"):
            check_status(r2r_client())

    with tab_settings:
        st.markdown("**Configuration Settings**")
        st.markdown("* Settings are read-only from here.")
        st.markdown("* To change them, update the server configuration and environment variables.")

        if st.button(label="Check configs", key="settings_check_button"):
            check_settings(r2r_client())
