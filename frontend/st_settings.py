import sys
import asyncio
import streamlit as st
from pathlib import Path
from app import load_client
from r2r import R2RException

backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_dir))

from settings import SystemHandler

# Create and set a persistent event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async(coro):
    return loop.run_until_complete(coro)

st.title("⚙️ Settings & System Information")

client = load_client()
system_handler = SystemHandler(client=client)

tab_health, tab_status, tab_settings, tab_logs = st.tabs(["Health", "Status", "Settings", "Logs"])

with tab_health:
    st.markdown("**Health Check**")
    health_btn = st.button("Check health")
    if health_btn:
        try:
            message = run_async(system_handler.health())
            st.success(f"Service Status: {message}", icon="✅")
        except R2RException as r2re:
            st.error(f"Error checking health: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab_status:
    st.markdown("**Fetch System Status**")
    status_btn = st.button("Get Status")
    if status_btn:
        try:
            status = run_async(system_handler.status())
        
            start_time = status.get("start_time", "N/A")
            uptime_seconds = status.get("uptime_seconds", "N/A")
            cpu_usage = status.get("cpu_usage", "N/A")
            memory_usage = status.get("memory_usage", "N/A")

            st.write(f"**Start Time:** {start_time}")
            st.write(f"**Uptime:** {uptime_seconds}")
            st.write(f"**CPU Usage:** {cpu_usage}")
            st.write(f"**Memory Usage:** {memory_usage}")
            
        except R2RException as r2re:
            st.error(f"Error fetching status: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

with tab_settings:
    st.markdown("**Configuration Settings**")
    st.markdown("Settings are read-only from here. To change them, update the server configuration and environment variables and restart the backend.")
    
    config_btn = st.button("Check configs")
    if config_btn:
        try:
            settings = run_async(system_handler.settings())
            if settings and isinstance(settings, dict):
                st.write("**R2R Backend Settings:**")
                
                # Option 1: Display entire JSON at once
                # st.json(settings)
                
                # Option 2: Display top-level keys in separate expanders (comment out st.json above if using this)
                for key, value in settings.items():
                    with st.expander(key, expanded=False):
                        # If the value is a dict, show it as json
                        if isinstance(value, dict):
                            st.json(value)
                        else:
                            st.write(value)
                
            else:
                st.info("No backend settings found.")
        except R2RException as r2re:
            st.error(f"Error fetching settings: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
with tab_logs:
    st.markdown("View the latest logs from the R2R service.")

    with st.form("log_form"):
        limit = st.number_input("Number of logs to fetch", min_value=10, max_value=100, value=100, step=10)
        offset = st.number_input("Offset (starting log)", min_value=0, value=0, step=10)
        submit_logs = st.form_submit_button("Fetch Logs")

    if submit_logs:
        try:
            logs = run_async(system_handler.logs(offset=offset, limit=limit))
            if logs:
                st.write(f"Displaying {len(logs)} logs (offset {offset}, limit {limit}):")
                st.json(logs)
            else:
                st.info("No logs found.")
        except R2RException as r2re:
            st.error(f"Error fetching logs: {r2re}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")