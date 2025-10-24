# python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --tp-size 8 --ep-size 8

MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite"
MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/IA/zhaoxiaoyu17/multimodal/deploy/flash_omini_1019_quant_moe_mlp_23_24_25_26_27/1/flash_omini_1019_quant_moe_mlp_23_24_25_26_27"
MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/IA/xiaobin14/sglang/longcat_omni"
# MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/flash_oss/flash_oss_fp8"
# MODEL_PATH='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/lisong39/ares_dpo/4_code_and_omni/ares_product_eval/moe26b/exp/moe26b_text_image_2508_speech_video_e7_e9_merge_add_image_v0_duotu_and_s_stage34_aito/ckpt/754/actor/iter_0000754/format_hf'
# MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/IA/xiaobin14/sglang/new_oss_bf16'
MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/IA/xiaobin14/sglang/new_oss_fp8'
# MODEL_PATH='/mnt/hdfs/zw04mlnn01/checkpoint/longcat-s/longcat-omni/Longcat-omni-univit0d6B-moe-26B-559B-2508/stage-RL/e7e9merge_dpo_1019/format_hf'
# MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/IA/xiaobin14/sglang/new_oss_fp16_2'
# MODEL_PATH/config.json
"""
"architectures": [
    "LongcatFlashOmniForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_longcat_flash.LongcatFlashConfig",
    "AutoModel": "modeling_longcat_flash.LongcatFlashModel",
    "AutoModelForCausalLM": "modeling_longcat_flash.LongcatFlashForCausalLM"
  },
"""
"""
need configuration_longcat_flash.py modeling_longcat_flash.py
"""
import os
import sys
import json
import asyncio

import torch
from transformers import AutoTokenizer
from safetensors import safe_open
import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge

# 加载embedding权重
def load_embedding_weights(model_path):
    # 查找safetensors文件
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    # 从第一个文件加载embedding权重
    safetensor_file = os.path.join(model_path, safetensor_files[0])
    with safe_open(safetensor_file, framework="pt") as f:
        # 尝试常见的embedding层名称
        possible_names = [
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.embeddings.word_embeddings.weight",
            "transformer.wte.weight"
        ]

        for name in possible_names:
            if name in f.keys():
                return f.get_tensor(name)

        # 如果都没找到，打印可用的keys
        print(f"Available keys: {list(f.keys())}")
        raise KeyError("Could not find embedding weights")

# 生成input_ids和input_embeds的函数
def generate_input_ids_and_embeds(prompts, tokenizer, embedding_layer):
    all_input_ids = []
    all_input_embeds = []

    for prompt in prompts:
        # 生成input_ids
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # 生成input_embeds
        with torch.no_grad():
            input_embeds = embedding_layer(input_ids)

        all_input_ids.append(input_ids.squeeze().tolist())
        all_input_embeds.append(input_embeds.squeeze())

    return all_input_ids, all_input_embeds


async def main(rank):
    prompts = [
        # "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "USER:The capital of France is\nVOICE ASSISTANT:",
        "USER:请将一个关于狗的笑话,尽量长一点,输出语音:\nVOICE ASSISTANT:",
        # "The future of AI is",
        # "please introduce yourself."
    ]

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载embedding权重并创建embedding层
    embedding_weights = load_embedding_weights(MODEL_PATH)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

    # 生成所有prompts的input_ids和input_embeds
    input_ids_list, input_embeds_list = generate_input_ids_and_embeds(prompts, tokenizer, embedding_layer)

    # 为了兼容原代码，我们使用第一个prompt的结果
    prompt=prompts[0]
    input_ids = input_ids_list[0]
    input_embeds = input_embeds_list[0]
    print(f"Generated input_ids shape: {len(input_ids)}")
    print(f"Generated input_embeds shape: {input_embeds.shape}")
    print(f"Input_ids: {input_ids}")
    print(f"Input_embeds: {input_embeds}")
    input_embeds = input_embeds.tolist()

    file = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hdp-hldy/hadoop-scale-llm/caowengang/baseline/case_1_base.pt"
    # file = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hdp-hldy/hadoop-scale-llm/ycz/workspace/release/omni_infer/input_embedding_no_voice_assisant.pt"
    input_embeds = torch.load(file).to(torch.float32).tolist()
    sampling_params = {"max_new_tokens": 256, "ignore_eos":True, "top_k": 1} # "temperature": 0.2, "top_p": 0.1, 

    OMNI_EXTRA_CONFIG = {
        "architectures": [
            "LongcatFlashOmniForCausalLM"
        ],
        "onmi_extra_info": {
            "hf_path" : MODEL_PATH, 
            "num_multi_ids": 4,
            "audio_head_num": 4,
            "audio_vocab_size": 8224,
            "audio_embed_pt": MODEL_PATH + "/audio/audio_embeddings.pt",
            "audio_id_offset": 32,
            "audio_rep_penalty_window": 30,
            "text_rep_penalty_window": 30,
            "audio_repetition_penalty" : 1.1,
            "audio_output_layer_pt": MODEL_PATH + "/audio/audio_output_layers.pt",
            "has_proj": False,
        }
    }
    node_rank=int(rank)
    if node_rank == -1:
        llm = sgl.Engine(model_path=MODEL_PATH, tp_size=8, ep_size=8, disable_radix_cache=True, disable_overlap_schedule=True, cuda_graph_max_bs=16, trust_remote_code=True, json_model_override_args=json.dumps(OMNI_EXTRA_CONFIG), disable_cuda_graph=True)#, disable_cuda_graph=True)mem_fraction_static=0.5)
    else:
        MASTER_IP="33.253.71.146:5000"
        MASTER_IP="10.238.45.157:5000"
        llm = sgl.Engine(model_path=MODEL_PATH, tp_size=16, ep_size=16, nnodes=2, node_rank=node_rank, 
                        mem_fraction_static=0.95,
                        dist_init_addr=MASTER_IP, disable_radix_cache=True, disable_overlap_schedule=True, cuda_graph_max_bs=16, 
                        trust_remote_code=True, json_model_override_args=json.dumps(OMNI_EXTRA_CONFIG), disable_cuda_graph=True)
    # print(f"=======================================")
    # outputs = await llm.async_generate(prompt=prompt, sampling_params=sampling_params)
    # # for prompt, output in zip(prompts, outputs):
    # #     print(f"\nPrompt: {prompt}")
    # #     print(f"Generated text: {output['text']}")
    # print(f"{prompt}:{outputs}")
    # print(f"=======================================")
    # outputs = await llm.async_generate(prompt=prompt, sampling_params=sampling_params)
    # # for prompt, output in zip(prompts, outputs):
    # #     print(f"\nPrompt: {prompt}")
    # #     print(f"Generated text: {output['text']}")
    # print(f"{prompt}:{outputs}")
    for x in range(1):
        print(f"=======================================")
        outputs = await llm.async_generate(input_embeds=input_embeds, sampling_params=sampling_params)# input_ids=input_ids, 
        print(f"{prompt}:{outputs}")


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))