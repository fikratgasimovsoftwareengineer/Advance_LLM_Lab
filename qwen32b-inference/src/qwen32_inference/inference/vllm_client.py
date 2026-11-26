from openai import AsyncOpenAI
from qwen32_inference.config import ApplicationConfig
from collections.abc import AsyncIterator

class UserChatCompletion:
    
    def __init__(self, 
                 config:ApplicationConfig, 
                 client:AsyncOpenAI):
        
        self.model_name = config.model.model_name
        self.client = client
        
        
    async def set_chat_completion(self, user_request:str)->AsyncIterator[str]:
        
        response = await self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {
                    "role":"user",
                    "content":user_request
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
    
