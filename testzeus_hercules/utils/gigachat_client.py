from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.completion_usage import CompletionUsage
from autogen.oai.client import OpenAIClient
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Dict, Any
import random
import time

GigaChat_PRICING_1K = {
    "gigachat-max": 2.5,
    "gigachat-pro": 1.5,
    "gigachat-plus": 0.4,
    "gigachat": 0.2,
}


def _calculate_giga_cost(prompt_tokens, completion_tokens, model_name):
    cost = GigaChat_PRICING_1K[model_name.lower()]
    total = prompt_tokens + completion_tokens
    return (total / 1000) * cost


class GigaChatClient(OpenAIClient):

    def __init__(self, config: dict[str, Any], **kwargs):
        self._client = GigaChat(
            base_url=config.get("base_url", None),
            auth_url=config.get("auth_url", None),
            credentials=config["api_key"],
            verify_ssl_certs=False,
            model=config["model"],
            scope=config.get("scope", "GIGACHAT_API_CORP"),
            timeout=config.get("timeout", 60),
            # verbose=config["verbose"],
        )

    @staticmethod
    def get_usage(response) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        # ...  # pragma: no cover
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost,
            "model": response.model,
        }

    def create(self, params):
        gigachat_messages = []
        ask_token_count = 0
        for message in params["messages"]:
            if message["role"] == "system":
                if len(gigachat_messages) > 0 and gigachat_messages[-1].type == "system":
                    # System messages can't appear after an Assistant message, so use a UserMessage
                    gigachat_messages.append(HumanMessage(content=message["content"]))
                else:
                    gigachat_messages.append(SystemMessage(content=message["content"]))
            if message["role"] == "user":
                gigachat_messages.append(HumanMessage(content=message["content"]))
            if message["role"] == "assistant":
                gigachat_messages.append(AIMessage(content=message["content"], name=message["name"]))
            tokens = self._client.get_num_tokens(message["content"])
            ask_token_count = + tokens

        # can create my own data response class
        # here using SimpleNamespace for simplicity
        # as long as it adheres to the ModelClientResponseProtocol

        ai_response = self._client(gigachat_messages)
        answer = ai_response.content
        tokens = self._client.get_num_tokens(answer)
        answer_token = tokens

        message = ChatCompletionMessage(role="assistant", content=answer, function_call=None, tool_calls=None)
        choices = [Choice(finish_reason="stop", index=0, message=message)]

        response = ChatCompletion(
            id=str(random.randint(0, 1000)),
            model=params["model"],
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=ask_token_count,
                completion_tokens=answer_token,
                total_tokens=ask_token_count + answer_token,
            ),
            cost=_calculate_giga_cost(ask_token_count, answer_token, params["model"])
        )
        return response

    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        return response.cost
