## 2025-12-21 - [Legacy LangChain Imports]
**Learning:** The codebase uses legacy `langchain` imports (e.g. `langchain.chains`) which break in modern `pip install langchain` environments (0.3+).
**Action:** When running locally, ensure strict dependency version pinning matching `Dockerfile` or use a compatible environment. Do not assume `pip install langchain` works out of the box.
