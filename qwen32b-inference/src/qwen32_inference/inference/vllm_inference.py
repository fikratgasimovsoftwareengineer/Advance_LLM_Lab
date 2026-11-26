import os
import shlex

from qwen32_inference.config import ApplicationConfig, load_config_yaml


def build_vllm_command(config:ApplicationConfig)->list[str]:
    
    model = config.model
    inference = config.inference
    distributed = config.distributed
    
    command  = [
        "vllm",
        "serve",
        model.model_name,
        
        '--quantization',
        model.quantizazione_type,
        
        '--dtype',
        model.dtype,
        
        '--tensor-parallel-size',
        distributed.tensor_parallel_size,
        
        '--pipeline-parallel-size',
        distributed.pipeline_parallel_size,
        
        '--distributed-executor-backend',
        distributed.distributed_executor_backend,
        
        '--gpu-memory-utilization',
        inference.gpu_memory_utilization,
        
        '--max-model-len',
        inference.max_model_len,
        
        '--host',
        inference.host,
        
        '--port',
        inference.port,
        
        '--max-num-seqs',
        inference.max_num_seqs
    
    ]
    
    if inference.enable_prefix_caching==True:
        command.append(
                '--enable-prefix-caching')
    else:
        command.append(
            '--no-enable-prefix-caching')
        
    return [str(arg) for arg in command]


def start_vllm_server(command:list[str])->None:
    
    print('Starting VLMM serving ..', flush=True)
    print(shlex.join(command), flush=True)
    
    os.execvp(command[0], command)

def main():
    
    config = load_config_yaml('configs')
    build_vlmm = build_vllm_command(config)
    start_vllm_server(build_vlmm)
    
if __name__ == "__main__":
    main()
    
    