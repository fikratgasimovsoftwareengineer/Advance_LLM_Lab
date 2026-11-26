from pydantic import BaseModel,Field
from typing import Literal
from qwen32_inference.config import *
import yaml
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR

class ModelConfig(BaseModel):
    model_name:str
    quantizazione_type:str
    dtype:Literal["float16", "bfloat16", "float32"]
    
class InferenceConfig(BaseModel):
    gpu_memory_utilization:float = Field(ge=0,le=1)
    max_model_len:int = Field(ge=0)
    enable_prefix_caching:bool = Field(default=True)
    host:str
    port:int
    max_num_seqs:int=  Field(gt=0)
    
class DistributedConfig(BaseModel):
    tensor_parallel_size:int = Field(gt=0)
    pipeline_parallel_size:int = Field(gt=0)
    distributed_executor_backend:Literal["ray"]
    
class ApplicationConfig(BaseModel):
    model:ModelConfig
    inference:InferenceConfig
    distributed:DistributedConfig
    
    

def load_config_yaml(conf_dir:str
) -> ApplicationConfig:


    configuration = {}

    files = [
        "distributed.yaml",
        "inference.yaml",
        "model.yaml"
    ]
    
    file_path = CONFIG_DIR / conf_dir
   

    for filename in files:
        file_path_join = file_path / filename

        with file_path_join.open("r", encoding="utf-8") as read_file:

            data = yaml.safe_load(read_file)

            configuration.update(data)

    return ApplicationConfig.model_validate(configuration)