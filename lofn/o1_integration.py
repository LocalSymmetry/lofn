from typing import List, Dict, Any, Optional
from langchain.schema import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatGeneration,
    ChatResult,
)
from langchain.chat_models.base import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# This PrivateAttr is from pydantic < 2, often used with langchain's pydantic_v1
from langchain_core.pydantic_v1 import Field, PrivateAttr, root_validator
from openai import OpenAI  # openai>=1.0.0 style usage

def _decide_max_completion_tokens(level: str) -> int:
    """
    Convert 'low'/'medium'/'high' to total tokens for the o1 or o1-mini models.
    """
    lv = level.lower()
    if lv == "low":
        return 12000
    elif lv == "high":
        return 30000
    else:
        # treat 'medium' or anything else as default
        return 20000

class O1ChatOpenAI(BaseChatModel):
    """
    A custom ChatModel for openai≥1.0.0 "o1" or "o1-mini" usage, 
    with typed fields recognized by Pydantic, 
    plus a PrivateAttr for the actual OpenAI client.
    """
    # Pydantic fields with defaults. 
    model_name: str = Field("o1", description="Which o1 or o1-mini model to use")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    reasoning_level: str = Field("medium", description="low, medium, or high")
    debug: bool = Field(False, description="Whether to log debug info")
    # We'll compute this after parsing, so set a dummy default
    max_completion_tokens: int = Field(0, description="o1 total tokens for (reasoning+completion)")
    # A private attribute that won't be validated or included in self.dict()
    _client: Optional[OpenAI] = PrivateAttr(default=None)

    def __init__(self, model_name, openai_api_key, reasoning_level, debug):
        """
        Custom constructor. Pydantic's BaseModel expects 
        parameters as **kwargs to parse them as fields. 
        That means usage: O1ChatOpenAI(model_name="o1", reasoning_level="low", etc.)
        not O1ChatOpenAI("o1").
        """
        # Let pydantic parse all the declared fields via super().__init__
        super().__init__()

        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.reasoning_level = reasoning_level
        self.debug = debug
        # Now that fields are parsed, we can compute or set additional values:
        # e.g. set max_completion_tokens from reasoning_level
        self.max_completion_tokens = _decide_max_completion_tokens(self.reasoning_level)

        # Create a single OpenAI client 
        self._client = OpenAI(
            api_key=self.openai_api_key
        )



    def _convert_langchain_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        openai_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                # doc suggests 'developer' role for o1, but 'system' is typically fine
                openai_messages.append({"role": "developer", "content": m.content})
            elif isinstance(m, HumanMessage):
                openai_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                openai_messages.append({"role": "assistant", "content": m.content})
            else:
                openai_messages.append({"role": "user", "content": m.content})
        return openai_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        # We'll pass max_completion_tokens plus any user overrides
        final_kwargs = {"max_completion_tokens": self.max_completion_tokens}
        final_kwargs.update(kwargs)

        openai_messages = self._convert_langchain_messages(messages)

        last_err = None
   
        try:
            if self.debug:
                print(f"[O1ChatOpenAI] Attempt; "
                      f"model={self.model_name}; "
                      f"max_completion_tokens={self.max_completion_tokens}; "
                      f"kwargs={kwargs}")

            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                **final_kwargs
            )
            if self.debug:
                print("Completion level completed without error")
                print(completion)
            text = completion.choices[0].message.content
            if self.debug:
                print("Text level completed without error")
                print(text)
            gen = ChatGeneration(message=AIMessage(content=text))
            if self.debug:
                print("Gen level completed without error")
                print(gen)
            return ChatResult(generations=[gen])
        except Exception as e:
            last_err = e
            if self.debug:
                print(f"[O1ChatOpenAI] Attempt {attempt+1} -> {e}")


    @property
    def _llm_type(self) -> str:
        return "o1-chat"

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        text = "".join(m.content for m in messages)
        return len(text.split())