import httpx

from openai import AsyncOpenAI
from fastapi.testclient import TestClient

from qwen32_inference.config import load_config_yaml

from qwen32_inference.api.server_restapi import RestAPI

from .test_llm_per_mock import mock_vllm_server

def test_rest_api():

    config = load_config_yaml("configs")

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                mock_vllm_server
            )
        )
    )

    rest_api = RestAPI(config, client)

    with TestClient(rest_api.app) as test_client:

        response = test_client.post(
            "/get_human_request",
            json={
                "user_request": "Ciao, come stai?"
            }
        )

        assert response.status_code == 200

        assert response.text == "Ciao, come stai?"

        print(response.text)

        print("REST API integration test passed successfully")


if __name__ == "__main__":

    test_rest_api()