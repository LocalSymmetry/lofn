import sys
import contextlib
import queue
import logging
from typing import Any, Dict, Optional, Generator

logger = logging.getLogger(__name__)

class SessionState(dict):
    """Mock session state allowing attribute access."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            # Return None or behave like defaultdict if needed, but standard behavior raises AttributeError
            # However, Streamlit session state usually returns None? No, it raises KeyError/AttributeError.
            # But let's be safe.
            raise AttributeError(f"'SessionState' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

class MockStatus:
    def __init__(self, label: str, queue_obj: queue.Queue):
        self.label = label
        self.queue = queue_obj
        self.state = "running"
        self._log("status", label, state="running")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        state = "error" if exc_type else "complete"
        self._log("status", self.label, state=state)

    def write(self, content: str):
        self._log("write", str(content))

    def update(self, label: str = None, state: str = None):
        if label:
            self.label = label
        if state:
            self.state = state
        self._log("status", self.label, state=self.state)

    def _log(self, type_: str, content: str, state: str = None):
        msg = {"type": type_, "content": str(content)}
        if self.label:
            msg["label"] = self.label
        if state:
            msg["state"] = state
        self.queue.put(msg)

class StreamlitMock:
    def __init__(self):
        self.queue = queue.Queue()
        self.session_state = SessionState()
        self._secrets = {}

    def write(self, *args, **kwargs):
        # Convert args to string similar to how streamlit does
        content = " ".join(map(str, args))
        self.queue.put({"type": "info", "content": content})

    def error(self, *args, **kwargs):
        content = " ".join(map(str, args))
        self.queue.put({"type": "error", "content": content})

    def success(self, *args, **kwargs):
        content = " ".join(map(str, args))
        self.queue.put({"type": "success", "content": content})

    def info(self, *args, **kwargs):
        content = " ".join(map(str, args))
        self.queue.put({"type": "info", "content": content})

    def warning(self, *args, **kwargs):
        content = " ".join(map(str, args))
        self.queue.put({"type": "warning", "content": content})

    def status(self, label: str, expanded: bool = False):
        return MockStatus(label, self.queue)

    @property
    def secrets(self):
        return self._secrets

    def spinner(self, text: str):
        self.write(f"Spinner: {text}")
        return contextlib.nullcontext()

    # Mock other potential calls to avoid crashes
    def markdown(self, body, unsafe_allow_html=False):
        self.write(body)

    def code(self, body, language="python"):
        self.write(f"Code ({language}):\n{body}")

    def json(self, body, expanded=True):
        import json
        self.write(json.dumps(body, indent=2, default=str))

@contextlib.contextmanager
def patch_streamlit(mock_instance: StreamlitMock):
    """Context manager to temporarily patch the streamlit module at runtime."""
    import streamlit as real_st

    # Attributes to patch
    methods = [
        'write', 'error', 'success', 'info', 'warning',
        'status', 'spinner', 'markdown', 'code', 'json'
    ]

    backup = {}

    # 1. Patch methods
    for method in methods:
        if hasattr(real_st, method):
            backup[method] = getattr(real_st, method)
            setattr(real_st, method, getattr(mock_instance, method))

    # 2. Patch session_state
    # session_state is a property on the module in recent versions?
    # Or an object. Let's try to overwrite it.
    if hasattr(real_st, 'session_state'):
        backup['session_state'] = real_st.session_state
        # We replace the object itself.
        # Note: If other modules did `from streamlit import session_state`, this won't update their reference.
        # But `llm_integration` does `import streamlit as st` and `st.session_state`. So this works.
        real_st.session_state = mock_instance.session_state

    try:
        yield
    finally:
        # Restore
        for method, original in backup.items():
            setattr(real_st, method, original)
