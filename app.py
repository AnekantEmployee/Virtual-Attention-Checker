import warnings
import streamlit as st
from views import ScreenshotManager


warnings.filterwarnings(
    "ignore", category=UserWarning, message="SymbolDatabase.GetPrototype.*"
)


st.title("Virtual Attention Checker")

# Initialize session state variables
if "manager" not in st.session_state:
    st.session_state.manager = ScreenshotManager(interval=1)
    st.session_state.is_running = False  # Initialize the running state


def start_monitoring():
    st.session_state.manager.start()
    st.session_state.is_running = True


def stop_monitoring():
    st.session_state.manager.stop()
    st.session_state.is_running = False


# Create columns for better button layout
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.is_running:
        if st.button("Start Monitoring", type="primary"):
            start_monitoring()
            st.rerun()

with col2:
    if st.session_state.is_running:
        if st.button("Stop Monitoring", type="secondary"):
            stop_monitoring()
            st.rerun()

# Status indicator
status_placeholder = st.empty()
if st.session_state.is_running:
    status_placeholder.success("Monitoring is active")
else:
    status_placeholder.info("Monitoring is stopped")

# You can add other UI elements or information here
st.write("Click 'Start Monitoring' to begin checking attention.")
