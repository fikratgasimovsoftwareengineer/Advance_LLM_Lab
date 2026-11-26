import json
import httpx


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


def mock_vllm_server(request: httpx.Request) -> httpx.Response:

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