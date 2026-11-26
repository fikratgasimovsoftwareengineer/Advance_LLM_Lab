from fastapi import FastAPI
from qwen32_inference.configs.i_rest_api import UtenteRequest
from dataclasses import dataclass
from fastapi.testclient import TestClient

@dataclass
class RestAPI:
    def __init__(self):
        self.app= FastAPI(
            title="Distributed LLM Inference API",
            version="0.1.0"
        )
        self.app.add_api_route(
            '/get_human_request',
            self.handle_request,
            methods = ["POST"]
        )
        
    
    async def handle_request(self, request:UtenteRequest):
        
        return {
            "user_message":request.user_request
        }
        
        
restapi = RestAPI()
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
        

        

