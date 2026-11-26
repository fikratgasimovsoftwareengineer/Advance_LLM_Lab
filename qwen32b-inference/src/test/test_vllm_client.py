import asyncio
import json
import httpx

from openai import AsyncOpenAI

from qwen32_inference.config import load_config_yaml

from qwen32_inference.inference.vllm_client import (
    UserChatCompletion
)


class MockStreamingResponse(httpx.AsyncByteStream):

    async def __aiter__(self):

        fragments = ["Ciao", ",", " come", " stai?"]

        for fragment in fragments:

            chunk = {
                "id": "test-001",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "Qwen/Qwen3-32B-AWQ",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": fragment
                        },
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"


def mock_vllm_server(request: httpx.Request):

    payload = json.loads(request.content)

    assert request.method == "POST"
    assert request.url.path == "/v1/chat/completions"
    assert payload["stream"] is True

    return httpx.Response(
        status_code=200,
        headers={
            "Content-Type": "text/event-stream"
        },
        stream=MockStreamingResponse()
    )


async def test_chat_completion():

    config = load_config_yaml("configs")

    transport = httpx.MockTransport(mock_vllm_server)

    async with AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        http_client=httpx.AsyncClient(
            transport=transport
        )
    ) as client:

        chat = UserChatCompletion(config, client)

        fragments = []

        async for content in chat.set_chat_completion(
            "Ciao, come stai?"
        ):

            print(content, end="", flush=True)

            fragments.append(content)

        assert fragments == [
            "Ciao", ",", " come", " stai?"
        ]

        assert "".join(fragments) == "Ciao, come stai?"

        print("\nStreaming test passed successfully")


if __name__ == "__main__":

    asyncio.run(test_chat_completion())