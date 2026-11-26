from qwen32_inference.inference.vllm_inference import build_vllm_command
from qwen32_inference.config import load_config_yaml
import shlex


def test_build_vllm():
    
    config = load_config_yaml('configs')
    
    vllm_params = build_vllm_command(config)
    
    assert vllm_params[:3] == [
        "vllm",
        "serve",
        "Qwen/Qwen3-32B-AWQ"
    ]
    assert vllm_params[vllm_params.index('--tensor-parallel-size')]=='--tensor-parallel-size'
    
    assert vllm_params[vllm_params.index('--tensor-parallel-size') + 1]=='2'
    
    assert vllm_params[vllm_params.index('--pipeline-parallel-size')+1]=='1'
    
    assert vllm_params[vllm_params.index('--gpu-memory-utilization')+1]== '0.9'
    
    assert vllm_params[vllm_params.index('--max-model-len')+1]=='4096'
    
    assert '--enable-prefix-caching' in vllm_params
    print("\n Generated Command")
    print(shlex.join(vllm_params))
    
    print("\nvLLM configuration test passed successfully")


def test_disable_prefix():
    
    config = load_config_yaml('configs')
    config.inference.enable_prefix_caching = 'False'   
    vllm_params = build_vllm_command(config)
    assert "--no-enable-prefix-caching" in vllm_params
    assert "--enable-prefix-caching" not in vllm_params


if __name__ == "__main__":

    test_build_vllm()
    test_disable_prefix()