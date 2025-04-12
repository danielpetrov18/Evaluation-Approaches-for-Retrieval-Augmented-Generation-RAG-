"""Simple file for separating the GUI from the logic as much as possible."""

# pylint: disable=W0718 -> disable too-broad of an exception

import datetime
import streamlit as st
from streamlit.errors import Error
from r2r import R2RException, R2RClient

def check_health(client: R2RClient):
    """Check health. Should return OK."""
    try:
        message = client.system.health().results.message
        st.success(f"Service status: {message}")
    except R2RException as r2re:
        st.error(f"Error checking health: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def check_status(client: R2RClient):
    """Check status. Gives information like uptime and memory usage."""
    try:
        with st.spinner(text="Fetching system status...", show_time=True):
            status = client.system.status().results

            # Calculate deltas to represent changes
            if 'previous_cpu_usage' not in st.session_state:
                # First run, no delta to display
                st.session_state['previous_cpu_usage'] = status.cpu_usage
                st.session_state['previous_memory_usage'] = status.memory_usage
                cpu_delta = None
                memory_delta = None
            else:
                # Calculate delta from previous values
                cpu_delta = status.cpu_usage - st.session_state['previous_cpu_usage']
                memory_delta = status.memory_usage - st.session_state['previous_memory_usage']
                st.session_state['previous_cpu_usage'] = status.cpu_usage
                st.session_state['previous_memory_usage'] = status.memory_usage

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{status.cpu_usage}%",
                delta=f"{cpu_delta:.1f}%" if cpu_delta is not None else None
            )
            st.metric(
                label="Memory Usage",
                value=f"{status.memory_usage}%",
                delta=f"{memory_delta:.1f}%" if memory_delta is not None else None
            )

        with col2:
            st.markdown("### Time Information")
            st.markdown(f"**Start Time:** {status.start_time}")

            uptime_formatted = _format_uptime(status.uptime_seconds)
            st.markdown(f"**Uptime:** {uptime_formatted}")

        st.caption(f"Last updated: {_get_current_time()}")
    except R2RException as r2re:
        st.error(f"Error fetching status: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

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
        st.error(f"Error fetching settings: {r2re.message}")
    except Error as e:
        st.error(f"Unexpected streamlit error: {str(e)}")
    except Exception as exc:
        st.error(f"Unexpected error: {str(exc)}")

def _format_uptime(seconds: float) -> str:
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

def _get_current_time() -> str:
    """Return current time formatted nicely."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
