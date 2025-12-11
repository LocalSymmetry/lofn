import contextlib
import logging

logger = logging.getLogger(__name__)

class StreamlitShim:
    def __init__(self):
        self.session_state = {}

    def write(self, *args, **kwargs):
        logger.info(f"st.write: {args} {kwargs}")

    def error(self, *args, **kwargs):
        logger.error(f"st.error: {args} {kwargs}")

    def success(self, *args, **kwargs):
        logger.info(f"st.success: {args} {kwargs}")

    def info(self, *args, **kwargs):
        logger.info(f"st.info: {args} {kwargs}")

    def warning(self, *args, **kwargs):
        logger.warning(f"st.warning: {args} {kwargs}")

    def markdown(self, *args, **kwargs):
        logger.info(f"st.markdown: {args} {kwargs}")

    def code(self, *args, **kwargs):
        logger.info(f"st.code: {args} {kwargs}")

    def json(self, *args, **kwargs):
        logger.info(f"st.json: {args} {kwargs}")

    @contextlib.contextmanager
    def status(self, label, **kwargs):
        logger.info(f"st.status: {label}")
        yield self

    def update(self, *args, **kwargs):
        pass

    def cache_data(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def sidebar(self, *args, **kwargs):
        return self # Mock sidebar objects

    def expander(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        # Default fallback for any other st.X call
        def mock(*args, **kwargs):
            logger.debug(f"st.{name} called with {args} {kwargs}")
            return self
        return mock

# Singleton instance
st = StreamlitShim()
