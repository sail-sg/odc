from dataclasses import dataclass

@dataclass
class Qwen2ModelConfig:
    num_layers: int
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int


@dataclass
class FlatParamSizes:
    layer: int
    embedding: int
    lm_head: int
    total: int


def qwen2_param_size(conf: Qwen2ModelConfig) -> FlatParamSizes:
    embedding = conf.vocab_size * conf.hidden_size
    lm_head = conf.hidden_size * conf.vocab_size
    layer_param_size = qwen2_param_size_per_layer(conf)
    norm = conf.hidden_size * 2
    total = layer_param_size * conf.num_layers + lm_head + embedding + norm
    return FlatParamSizes(
        layer=layer_param_size,
        embedding=embedding,
        lm_head=lm_head,
        total=total,
    )


# def qwen2_lmhead_size(conf: Qwen2ModelConfig) -> int:
#     return conf.hidden_size * conf.vocab_size


def qwen2_param_size_per_layer(conf: Qwen2ModelConfig) -> int:
    num_groups = conf.num_heads // conf.num_kv_heads
    assert num_groups > 0
    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/model.safetensors
    input_layernorm = conf.hidden_size
    post_attention_layernorm = conf.hidden_size
    q_proj = conf.hidden_size * conf.hidden_size
    head_size = conf.hidden_size // num_groups
    k_proj = conf.hidden_size * head_size
    v_proj = conf.hidden_size * head_size
    q_proj_bias = conf.hidden_size
    k_proj_bias = head_size
    v_proj_bias = head_size
    o_proj = conf.hidden_size * conf.hidden_size
    # print(f"QKV: {q_proj + k_proj + v_proj + q_proj_bias + k_proj_bias + v_proj_bias + o_proj}")
    # SwiGLU
    reduced_size = conf.intermediate_size  # Not reduced actually
    up_proj = conf.hidden_size * reduced_size
    gate_proj = conf.hidden_size * reduced_size
    down_proj = reduced_size * conf.hidden_size
    # print(f"FFN: {up_proj + gate_proj + down_proj}")
    # assert up_proj + down_proj == conf.intermediate_size * conf.hidden_size * 2, f"{up_proj + down_proj} != {conf.intermediate_size * conf.hidden_size * 2}"
    return q_proj + k_proj + v_proj + q_proj_bias + k_proj_bias + v_proj_bias + o_proj + up_proj + gate_proj + down_proj + input_layernorm + post_attention_layernorm


qwen25_math_1_5B_config = Qwen2ModelConfig(
    num_layers=28,
    vocab_size=151936,
    hidden_size=1536,
    num_heads=12,
    num_kv_heads=2,
    intermediate_size=8960,
)

qwen25_math_7B_config = Qwen2ModelConfig(
    num_layers=28,
    vocab_size=152064,
    hidden_size=3584,
    num_heads=28,
    num_kv_heads=4,
    intermediate_size=18944,
)

qwen25_math_14B_config = Qwen2ModelConfig(
    num_layers=48,
    vocab_size=152064,
    hidden_size=5120,
    num_heads=40,
    num_kv_heads=8,
    intermediate_size=13824,
)

qwen25_math_32B_config = Qwen2ModelConfig(
    num_layers=64,
    vocab_size=152064,
    hidden_size=5120,
    num_heads=40,
    num_kv_heads=8,
    intermediate_size=27648,
)

qwen25_1_5B_size = qwen2_param_size(qwen25_math_1_5B_config)
print(f"1.5B lm_head: {qwen25_1_5B_size.lm_head}")
print(f"1.5B layer: {qwen25_1_5B_size.layer}")
print(f"1.5B total: {qwen25_1_5B_size.total}")

qwen25_7B_size = qwen2_param_size(qwen25_math_7B_config)
print(f"7B lm_head: {qwen25_7B_size.lm_head}")
print(f"7B layer: {qwen25_7B_size.layer}")
print(f"7B total: {qwen25_7B_size.total}")

qwen25_14B_size = qwen2_param_size(qwen25_math_14B_config)
print(f"14B lm_head: {qwen25_14B_size.lm_head}")
print(f"14B layer: {qwen25_14B_size.layer}")
print(f"14B total: {qwen25_14B_size.total}")

qwen25_32B_size = qwen2_param_size(qwen25_math_32B_config)
print(f"32B lm_head: {qwen25_32B_size.lm_head}")
print(f"32B layer: {qwen25_32B_size.layer}")
print(f"32B total: {qwen25_32B_size.total}")

single_host_sizes = [qwen25_1_5B_size, qwen25_7B_size]
rs_input_sizes = []
for size in single_host_sizes:
    rs_input_sizes.append(size.layer)
    rs_input_sizes.append(size.embedding)
print(f"rs_input_sizes: {sorted(rs_input_sizes)}")

