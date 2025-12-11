import logging
from config import Config

logger = logging.getLogger(__name__)

try:
    from google import genai
except ImportError:
    genai = None

def run_deep_research_stream(input_text: str, agent_name: str = "deep-research-pro-preview-12-2025"):
    """
    Executes a Deep Research task using Google Gemini 3 Pro agent.
    Yields dicts with keys 'type' and 'content'.
    Types: 'info', 'text_delta', 'thought', 'complete', 'error'.
    """
    if genai is None:
        yield {"type": "error", "content": "google.genai library is not installed or failed to load."}
        return

    if not Config.GOOGLE_API_KEY:
        yield {"type": "error", "content": "GOOGLE_API_KEY is not set."}
        return

    try:
        client = genai.Client(api_key=Config.GOOGLE_API_KEY)

        # Check if interactions attribute exists (it's experimental)
        if not hasattr(client, 'interactions'):
             yield {"type": "error", "content": "The installed google.genai library does not support 'interactions'. Please update the library."}
             return

        stream = client.interactions.create(
            input=input_text,
            agent=agent_name,
            background=True,
            stream=True,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "auto"
            }
        )

        interaction_id = None

        for chunk in stream:
            if hasattr(chunk, 'event_type'):
                if chunk.event_type == "interaction.start":
                    interaction_id = chunk.interaction.id
                    yield {"type": "info", "content": f"Research started. ID: {interaction_id}"}

                elif chunk.event_type == "content.delta":
                    if chunk.delta.type == "text":
                        yield {"type": "text_delta", "content": chunk.delta.text}
                    elif chunk.delta.type == "thought_summary":
                        content_text = ""
                        if hasattr(chunk.delta, 'content') and hasattr(chunk.delta.content, 'text'):
                            content_text = chunk.delta.content.text
                        yield {"type": "thought", "content": content_text}

                elif chunk.event_type == "interaction.complete":
                    yield {"type": "complete", "content": "Research Complete"}

                elif chunk.event_type == "error":
                     msg = chunk.error.message if hasattr(chunk.error, 'message') else str(chunk.error)
                     yield {"type": "error", "content": f"Error event: {msg}"}
            else:
                 yield {"type": "debug", "content": str(chunk)}

    except Exception as e:
        yield {"type": "error", "content": f"Exception during Deep Research: {str(e)}"}
