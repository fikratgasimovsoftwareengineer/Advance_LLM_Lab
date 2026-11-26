from pydantic import BaseModel

class UtenteRequest(BaseModel):
    user_request:str