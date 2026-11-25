from qwen32_inference.config import load_config_yaml


def test_load_configuration():

    config = load_config_yaml("configs")

    assert config.model.model_name == "Qwen/Qwen3-32B-AWQ"
    assert config.distributed.tensor_parallel_size == 2
    assert config.inference.enable_prexic_caching is True
    assert config.inference.gpu_memory_utilizzation == 0.90

    print(config.model_dump_json(indent=4))
    print("Configuration test passed successfully")


if __name__ == "__main__":
    test_load_configuration()