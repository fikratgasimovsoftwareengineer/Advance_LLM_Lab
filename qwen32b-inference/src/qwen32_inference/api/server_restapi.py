from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from contextlib import asynccontextmanager

from qwen32_inference.inference.vllm_client import UserChatCompletion
from qwen32_inference.config import load_config_yaml, ApplicationConfig
from qwen32_inference.configs.i_rest_api import UtenteRequest

@dataclass
class RestAPI:
    
    def __init__(self, 
                 config:ApplicationConfig,
                 client:AsyncOpenAI):
        
        self.app= FastAPI(
            title="Distributed LLM Inference API",
            version="0.1.0",
            ## function
            lifespan=self.lifespan
        )
        self.app.add_api_route(
            '/get_human_request',
            self.handle_request,
            methods = ["POST"]
        )
        
        ## INIT USER CHAT COMPLETIION ##
        self.chat_completion = UserChatCompletion(config, client)
        self.client = client
        
        
    @asynccontextmanager
    async def lifespan(self, app:FastAPI):
        print(f"Application startup")
        try:
            yield
        finally:
            await self.client.close()
            
    async def handle_request(self, request:UtenteRequest):
        
        return StreamingResponse(
            self.chat_completion.set_chat_completion(request.user_request),
            media_type="text/plain; charset=utf-8",
            headers={
                "X-Accel-Buffering":"no",
                "Cache-Control":"no-cache"
            }
        )
        
configs = load_config_yaml('configs')

api_openai_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="EMPTY"
)
restapi = RestAPI(configs,api_openai_client)
app = restapi.app

""" def main():
    app = RestAPI()
    client = TestClient(app.app)

    response = client.post(
        "/get_human_request", json ={"user_request": "Ciao, test in debug!"})
    
    print(response.status_code)
    print(response.json())
if __name__ == "__main__":
    main() """
        

        

