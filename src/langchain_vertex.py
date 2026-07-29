import os
import json
from typing import Any, List, Optional, Type, Union
from pydantic import BaseModel, Field, PrivateAttr
from google.genai import types
from google import genai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import Runnable


# 1. Define VertexAIChat Class (Internal genai.Client initialization)
class VertexAIChat(BaseChatModel):
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_key: Optional[str] = Field(default=None)
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is missing. Please add it to your .env file.")
        # Automatically initialize the working vertexai=True client internally
        self._client = genai.Client(vertexai=True, api_key=key)

    @property
    def _llm_type(self) -> str:
        return "vertex_ai_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_parts = []
        for m in messages:
            role = "User" if m.type == "human" else "Assistant"
            prompt_parts.append(f"{role}: {m.content}")

        prompt = "\n".join(prompt_parts)
        config_kwargs = {"temperature": self.temperature}
        if stop:
            config_kwargs["stop_sequences"] = stop

        response_schema = kwargs.get("response_schema")
        response_mime_type = kwargs.get("response_mime_type")
        if response_schema:
            config_kwargs["response_schema"] = response_schema
            config_kwargs["response_mime_type"] = response_mime_type or "application/json"
        elif response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

        config = types.GenerateContentConfig(**config_kwargs)

        res = self._client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )

        # 1. Extract token counts from Google GenAI response
        usage_info = getattr(res, "usage_metadata", None)
        input_tokens = usage_info.prompt_token_count if usage_info else 0
        output_tokens = usage_info.candidates_token_count if usage_info else 0
        total_tokens = usage_info.total_token_count if usage_info else 0

        # 2. Build response_metadata dictionary
        response_metadata = {
            "model_name": self.model,
            "token_usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }

        # 3. Build standardized usage_metadata dictionary
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        # 4. Pass metadata into AIMessage
        ai_message = AIMessage(
            content=res.text or "",
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        bound_model = self.bind(response_schema=schema, response_mime_type="application/json")

        def _parse(ai_message: AIMessage) -> Any:
            text = ai_message.content
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate_json(text)
            elif isinstance(schema, dict):
                return json.loads(text)
            else:
                try:
                    return schema.model_validate_json(text)
                except Exception:
                    return json.loads(text)

        if include_raw:
            def _parse_raw(ai_message: AIMessage) -> dict:
                return {
                    "raw": ai_message,
                    "parsed": _parse(ai_message),
                    "parsing_error": None,
                }

            return bound_model | _parse_raw
        else:
            return bound_model | _parse
