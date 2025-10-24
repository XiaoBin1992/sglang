# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import logging
from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.configs import LongcatFlashConfig
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from torch import Tensor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from torch.nn.utils.rnn import pad_sequence
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.models.longcat_flash import (
    LongcatFlashForCausalLM, LongcatFlashModel, LongcatFlashMLP
)
if torch.cuda.is_available():
    from sgl_kernel import (
        top_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

class SpecialTokens:
    def __init__(self, hf_path):
        self.AUDIO_SEMANTIC_EOS = 2
        self.EOS = 2
        self.CONT_TEXT_TO_END_MARKS = [491, 525, 1266, 1361, 986, 235, 237, 254, 5481]
        self.CONT_TEXT_END = 101
        self.CHAT_TEXT_FREE_END = 137
        self.CHAT_TEXT_FINALLY_END = 2
        self.TTS_TEXT_END = self.CONT_TEXT_END

        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        func_call_mark = "<longcat_tool_call>"
        func_call_ids = self.tokenizer.encode(func_call_mark)
        self.FUNC_CALL_START = func_call_ids[0]
        logger.info(f"\033[32m[{func_call_mark=}, {func_call_ids=}, {self.FUNC_CALL_START=}]\033[0m")

# SPT_KEY = "longcat_omni_spt"
__SpecialTokens_INSTANCE = None
def init_spt(hf_path) -> SpecialTokens:
    spec = SpecialTokens(hf_path)
    global __SpecialTokens_INSTANCE
    __SpecialTokens_INSTANCE = spec


def get_spt() -> SpecialTokens:
    global __SpecialTokens_INSTANCE
    return __SpecialTokens_INSTANCE

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class StateMachineInput:
    text_id: int = None
    semantic_id: int = None
    req_type: str = None


class StateEnum(Enum):
    INIT = auto()
    ABORT = auto()
    # Audio Text
    AT_ST0 = auto()
    CONT_ST0_FORCE_END = auto()
    AT_ST1 = auto()


class SmContext:
    def __init__(self):
        self.req_type: str = None
        self.to_abort: bool = False
        self.tf_end: bool = False
        self.step: int = -1

        self.rank = torch.cuda.current_device()


class StateBase:
    def on_enter(self, context: SmContext) -> None:
        pass

    def on_exit(self, context: SmContext) -> None:
        pass

    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        raise NotImplementedError("")


# State registry to maintain the mapping
_STATE_REGISTRY: Dict[StateEnum, type] = {}


def bind(state_type: StateEnum):
    def decorator(cls):
        _STATE_REGISTRY[state_type] = cls
        return cls

    return decorator


@bind(StateEnum.INIT)
class INIT_State(StateBase):
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        context.req_type = input.req_type
        assert input.req_type in ("cont", "chat")
        return StateEnum.AT_ST0


@bind(StateEnum.AT_ST0)
class AT_ST0_State(StateBase):
    def on_enter(self, context):
        context.step = 0

    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        context.step += 1
        if context.req_type == "cont":
            if input.text_id in get_spt().CONT_TEXT_TO_END_MARKS:
                # 遇到标点符号之后，强制插入一个结束符号
                return StateEnum.CONT_ST0_FORCE_END
        else:
            assert context.req_type == "chat"
            if input.text_id == get_spt().CHAT_TEXT_FREE_END:
                # 对于 CHAT，自由生成 Stage 0 的结束符号
                return StateEnum.AT_ST1
            elif input.text_id == get_spt().CHAT_TEXT_FINALLY_END:
                # 如果生成了全局结束符号，在生成完毕本轮 Audio 之后强制终止
                context.to_abort = True
                return StateEnum.AT_ST1


@bind(StateEnum.CONT_ST0_FORCE_END)
class CONT_ST0_FORCE_END_State(StateBase):
    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        # cont 只有一轮
        context.to_abort = True
        return StateEnum.AT_ST1


@bind(StateEnum.AT_ST1)
class AT_ST1_State(StateBase):
    def on_enter(self, context):
        context.step = 0

    def handle(self, input: StateMachineInput, context: SmContext) -> Optional[StateEnum]:
        context.step += 1
        if input.semantic_id == get_spt().AUDIO_SEMANTIC_EOS:
            if context.to_abort:
                return StateEnum.ABORT
            else:
                # 自动继续下一轮的生成
                return StateEnum.AT_ST0


@bind(StateEnum.ABORT)
class ABORT_State(StateBase):
    pass


# ============Main State Machine============
class StateMachine:
    def __init__(self):
        self.prev_state_enum: StateEnum = StateEnum.INIT
        self.cur_state_enum: StateEnum = StateEnum.INIT
        self.context = SmContext()
        self._states = {state_type: cls() for state_type, cls in _STATE_REGISTRY.items()}

    def transition(self, new_state_enum: StateEnum) -> bool:
        self._states[self.cur_state_enum].on_exit(self.context)
        self.prev_state_enum = self.cur_state_enum
        self.cur_state_enum = new_state_enum
        self._states[self.cur_state_enum].on_enter(self.context)

    def process(self, input: StateMachineInput) -> Tuple[StateEnum, StateEnum]:
        self.last_input = input
        next_state = self._states[self.cur_state_enum].handle(input, self.context)
        if next_state is not None:
            self.transition(next_state)
            return True
        return False

    def get_state(self) -> StateEnum:
        return self.cur_state_enum

    def get_step(self):
        return self.context.step

    def to_string(self):
        return f"prev: {self.prev_state_enum.name}, cur: {self.cur_state_enum.name}, last_input: {self.last_input.__dict__}, context: {self.context.__dict__}"



class AudioEmbedding(torch.nn.Module):
    def __init__(self, hidden_size, audio_vocab_size, audio_head_num, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_head_num = audio_head_num
        self.audio_embeddings = torch.nn.ModuleList(
            [
                VocabParallelEmbedding(self.audio_vocab_size, self.hidden_size, padding_size=1, enable_tp=False)
                for _ in range(audio_head_num)
            ]
        )
        self.dtype = dtype

    @torch.compile(dynamic=False)
    def fused_lookup(self, codecs: torch.Tensor):
        batch_dims = codecs.shape[:-1]
        combined_embedding = torch.zeros((*batch_dims, self.hidden_size), device="cuda", dtype=self.dtype)
        valid_mask = codecs >= 0
        for i in range(self.audio_head_num):
            head_mask = valid_mask[..., i]
            if head_mask.any():
                valid_ids = codecs[..., i][head_mask]
                embeddings = self.audio_embeddings[i](valid_ids)
                combined_embedding[head_mask] += embeddings
        return combined_embedding
    
class RepetitionPenaltyType(Enum):
    MULTIPLICATION = 0 # 乘法
    ADDITION = 1 # 加法
    EXPONENTIAL = 2 # 累乘


class AudioOutputLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        audio_vocab_size,
        audio_head_num,
        has_proj=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_head_num = audio_head_num
        self.audio_output_layers = torch.nn.ModuleList(
            [
                Linear(in_features=hidden_size, out_features=audio_vocab_size, bias=False)
                for _ in range(self.audio_head_num)
            ]
        )
        self.has_proj = has_proj
        if has_proj:
            self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.use_rq_codebook = True
        self.embedding = torch.nn.ModuleList(
            [torch.nn.Embedding(audio_vocab_size, hidden_size) for _ in range(self.audio_head_num - 1)]
        )

        self.attn_tp_rank = get_attention_tp_rank()
        self.step = 0

    def penalize_repetition(self, head_past_codecs, cur_audio_logits, audio_repetition_penalty, repetition_penalty_type=2):
        batch_size, seq_len, vocab_size = cur_audio_logits.shape
        assert seq_len == 1, f"{cur_audio_logits.shape=}"
        assert head_past_codecs.dim() == 2 and head_past_codecs.shape[0] == batch_size, f"{head_past_codecs.shape=}"

        if head_past_codecs.numel() == 0:
            return cur_audio_logits, torch.argmax(cur_audio_logits, dim=-1)

        valid_mask = (head_past_codecs >= 0) & (head_past_codecs < vocab_size)
        safe_codecs = head_past_codecs.masked_fill(~valid_mask, 0)
        audio_freq = F.one_hot(safe_codecs, vocab_size).float()  # (batch_size, seq_len, vocab_size)
        audio_freq = (audio_freq * valid_mask.unsqueeze(-1)).sum(1)  # (batch_size, vocab_size)

        # 确保 rep_penalty 的维度为 (batch_size, 1, 1)
        audio_repetition_penalty = audio_repetition_penalty.view(-1, 1, 1)  # (batch_size, 1, 1)
        if repetition_penalty_type == RepetitionPenaltyType.EXPONENTIAL:
            # 累乘
            audio_alpha = audio_repetition_penalty ** audio_freq.unsqueeze(1)  # (batch_size, 1, vocab_size)
            penalized_audio_logits = torch.where(
                cur_audio_logits < 0, cur_audio_logits * audio_alpha, cur_audio_logits / audio_alpha
            )
        elif repetition_penalty_type == RepetitionPenaltyType.MULTIPLICATION:
            # 乘法
            audio_alpha = torch.where(audio_freq.unsqueeze(1) > 0, audio_repetition_penalty, torch.tensor(1.0))
            penalized_audio_logits = torch.where(
                cur_audio_logits < 0, cur_audio_logits * audio_alpha, cur_audio_logits / audio_alpha
            )
        elif repetition_penalty_type == RepetitionPenaltyType.ADDITION:
            # 加法
            audio_alpha = torch.where(audio_freq.unsqueeze(1) > 0, audio_repetition_penalty-1, torch.tensor(0.0))
            penalized_audio_logits = cur_audio_logits - audio_alpha
        else:
            penalized_audio_logits = cur_audio_logits
        return penalized_audio_logits, torch.argmax(penalized_audio_logits, dim=-1)

    def batch_penalize_repetition(self, head_past_codecs, cur_audio_logits, audio_repetition_penalty, repetition_penalty_type):
        # print("repetition_penalty_type", repetition_penalty_type)
        if len(set(repetition_penalty_type)) == 1:
            # batch内所有惩罚类型一致，批量处理
            penalized_audio_logits, penalized_audio_id = self.penalize_repetition(
                head_past_codecs,cur_audio_logits,audio_repetition_penalty,repetition_penalty_type[0]
            )
        else :
            # print("split")
            # 分别处理
            penalized_audio_logits_list=[]
            penalized_audio_id_list=[]
            for index in range(cur_audio_logits.shape[0]):
                penalized_audio_logits_one, penalized_audio_id_one = self.penalize_repetition(
                    head_past_codecs=head_past_codecs[index,:].unsqueeze(0), #(b=1,len)
                    cur_audio_logits=cur_audio_logits[index,:,:].unsqueeze(0), #(b=1,1,v)
                    audio_repetition_penalty=audio_repetition_penalty[index].unsqueeze(0), #(b=1)
                    repetition_penalty_type=repetition_penalty_type[index],
                )
                penalized_audio_logits_list.append(penalized_audio_logits_one)
                penalized_audio_id_list.append(penalized_audio_id_one)
            # 拼接结果
            penalized_audio_logits = torch.cat(penalized_audio_logits_list, dim=0)
            penalized_audio_id = torch.cat(penalized_audio_id_list, dim=0)
        return penalized_audio_logits, penalized_audio_id
                
    def forward(self, lm_output, past_audio_ids, audio_repetition_penalty, repetition_penalty_type):
        batch_size, audio_head_num, max_seq_len = past_audio_ids.shape
        acc_output = lm_output
        raw_logits_list = []
        logits_list = []
        ids_list = []

        for head_idx in range(audio_head_num):
            if head_idx > 0:
                cur_embedding = self.embedding[head_idx - 1](penalized_audio_id)
                acc_output = acc_output + cur_embedding

            if self.has_proj:
                acc_output = self.linear(acc_output)

            audio_logits = self.audio_output_layers[head_idx](acc_output)
            raw_logits_list.append(audio_logits)
            head_past_ids = past_audio_ids[:, head_idx, :].long().cuda()
            penalized_audio_logits, penalized_audio_id = self.batch_penalize_repetition(
                head_past_codecs=head_past_ids,
                cur_audio_logits=audio_logits,
                audio_repetition_penalty=audio_repetition_penalty,
                repetition_penalty_type=repetition_penalty_type,
            )
            logits_list.append(penalized_audio_logits)
            ids_list.append(penalized_audio_id)

        self.step += 1
        return raw_logits_list, logits_list, ids_list


class LongcatFlashOmniForCausalLM(LongcatFlashForCausalLM):
    def __init__(
        self,
        config: LongcatFlashConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        # print(f"config:{config}")
        config_dict = config.onmi_extra_info
        # OMNI_EXTRA_CONFIG = os.environ["OMNI_EXTRA_CONFIG"]
        # config_dict = json.loads(OMNI_EXTRA_CONFIG)
        init_spt(config_dict["hf_path"])
        self.base_model = self.model
        self.hidden_size = self.base_model.embed_tokens.embedding_dim
        self.dtype = self.base_model.embed_tokens.weight.dtype
        self.audio_embed = AudioEmbedding(
            hidden_size=self.hidden_size,
            audio_head_num=config_dict["audio_head_num"],
            audio_vocab_size=config_dict["audio_vocab_size"],
            dtype=self.dtype,
        )
        self.attn_tp_rank = get_attention_tp_rank()
        self.audio_embed.load_state_dict(torch.load(config_dict["audio_embed_pt"]))
        self.audio_embed = self.audio_embed.cuda()

        self.sm_dict = {}
    
        self.audio_head_num = config_dict["audio_head_num"]
        self.hidden_size = self.base_model.embed_tokens.embedding_dim
        self.audio_output_layer = AudioOutputLayer(
            hidden_size=self.hidden_size,
            audio_vocab_size=config_dict["audio_vocab_size"],
            audio_head_num=self.audio_head_num,
            has_proj=config_dict["has_proj"],
        ).cuda()
        audio_output_layer_pt = torch.load(config_dict["audio_output_layer_pt"])
        self.audio_output_layer.load_state_dict(audio_output_layer_pt)
        self.audio_output_layer = self.audio_output_layer.to(torch.bfloat16)
        self.audio_id_offset = config_dict["audio_id_offset"]
        self.audio_rep_penalty_window = config_dict["audio_rep_penalty_window"]
        self.text_rep_penalty_window = config_dict["text_rep_penalty_window"]
        self.audio_repetition_penalty = config_dict["audio_repetition_penalty"]

    @torch.compile(dynamic=False)
    def masked_text_lookup(self, text_ids: torch.Tensor) -> torch.Tensor:
        valid_mask = text_ids >= 0
        safe_ids = text_ids.masked_fill(~valid_mask, 0)
        embeddings = self.base_model.embed_tokens(safe_ids).to(self.base_model.embed_tokens.weight.dtype)
        embeddings.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
        return embeddings

    def input_process_forward_decode(self,
        input_ids: Optional[Tensor] = None,
        input_multi_ids: Optional[Tensor] = None,
        batch_size: Optional[Tensor] = None,
        reqs: Optional[Tensor] = None,
    ):
        tp_num_tokens = input_ids.shape[0]
        input_multi_ids = torch.as_tensor(input_multi_ids, dtype=torch.int64, device=input_ids.device)
        input_multi_ids = input_multi_ids.reshape(tp_num_tokens, -1)
        assert tp_num_tokens == batch_size
        slices = [slice(i, i + 1) for i in range(batch_size)]

        def process_req(req_idx):
            req: Req = reqs[req_idx]
            sm: StateMachine = self.sm_dict[req.rid]
            slice = slices[req_idx]

            if sm.get_state() == StateEnum.AT_ST0:
                if sm.get_step() == 0:
                    # Decode 阶段的 AT_ST0 之前一定是 AT_ST1
                    # 此时需要抛弃 AT_ST1 当中无效的 input_ids
                    input_ids[slice].fill_(-1000012)

            if sm.get_state() == StateEnum.AT_ST1:
                if sm.get_step() == 0:
                    # QUESTION[zhaoxiaoyu17]: 这一步是不是不需要 mask 任何东西
                    # QUESTION[zhaoxiaoyu17]: 直接输入 Text: 101; Semantic + 3 Audio
                    pass
                else:
                    # Stage1 不需要文本
                    input_ids[slice].fill_(-1000021)

        for req_idx in range(len(reqs)):
            process_req(req_idx)

        text_emb = self.masked_text_lookup(input_ids)
        audio_emb = self.audio_embed.fused_lookup(input_multi_ids)
        hidden_states = audio_emb + text_emb

        return hidden_states


    def mask_except(self, logits, idx):
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, idx] = False
        logits[mask] = -torch.inf
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def sample(
        self,
        forward_batch: ForwardBatch,
        sample_func: Callable,
        text_logits_output: LogitsProcessorOutput,
    ):
        sample_hidden_states = text_logits_output.hidden_states
        # print(f"sample_hidden_states: {sample_hidden_states.shape}")
        sample_hidden_states = sample_hidden_states.reshape(forward_batch.batch_size, 1, sample_hidden_states.shape[-1])
        # logger.info(
        #     f"\033[33m[{forward_batch.forward_mode=}, {probe(output_hidden_states)=} ,{probe(sample_hidden_states)=}]\033[0m"
        # )

        # req = forward_batch.reqs[0]
        # req.output_extra_info = {}

        # forward_batch.temp_multi_ids = torch.full(
        #     (forward_batch.batch_size, self.audio_head_num),
        #     -999997,
        #     dtype=torch.int32,
        #     device=device,
        # )

        for i, req in enumerate(forward_batch.reqs):
            # print(f"xxx req:{req.rid} is_chunked:{req.is_chunked}")
            if "gen_state" not in req.aux_output_infos:
                req.aux_output_infos["gen_state"] = []
            if "audio_codes" not in req.aux_output_infos:
                req.aux_output_infos["audio_codes"] = []
            if "text_codes" not in req.aux_output_infos:
                req.aux_output_infos["text_codes"] = []

        raw_logits_list, panelized_logits_list, audio_ids_list = self.audio_forward(
            forward_batch=forward_batch,
            hidden_states=sample_hidden_states,
        )
        audio_ids = torch.concat(audio_ids_list, dim=0)
        audio_ids = audio_ids.reshape(self.audio_head_num, forward_batch.batch_size).transpose(0, 1)
        # 文本id计算
        past_output_text_ids = []
        topk_list = []
        topp_list=[]
        temperature=[]
        repetition_penalty=[]
        for req in forward_batch.reqs:            
            past_output_text_ids.append(req.aux_output_infos["text_codes"][-1*self.audio_rep_penalty_window:])
            topk_list.append(req.sampling_params.top_k)
            topp_list.append(req.sampling_params.top_p)
            temperature.append(req.sampling_params.temperature)
            repetition_penalty.append(req.sampling_params.repetition_penalty)
        # 文本采样及惩罚
        text_ids = self.sample_inner(
            next_token_logits=text_logits_output.next_token_logits,
            past_text_ids=past_output_text_ids,
            top_p=topp_list,
            top_k=topk_list,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        output_text_ids = text_ids.clone()
        decode_idx = []
        finish_rids = []
        def process_req(req_idx):
            req: Req = forward_batch.reqs[req_idx]
            if req.is_chunked > 0:
                return
            # decode first token
            if len(req.output_ids) == 0: #req.rid not in self.sm_dict:
                sm = StateMachine()
                self.sm_dict[req.rid] = sm
                # TODO check req_type="chat"
                sm_input = StateMachineInput(req_type="chat") # req_type=req.input_extra_infos[0]["req_type"]
                trans = sm.process(sm_input)
                if trans and self.attn_tp_rank == 0:
                    logger.info(f"\033[34m[[{req_idx=}, Extend finished. Rid {req.rid}] {sm.to_string()}]\033[0m")

            decode_idx.append(req_idx)
            sm: StateMachine = self.sm_dict[req.rid]

            real_text_id = text_ids[req_idx].item()


            if sm.get_state() == StateEnum.CONT_ST0_FORCE_END:
                real_text_id = get_spt().CONT_TEXT_END


            semantic_id = audio_ids_list[0][req_idx].squeeze()
            cur_state_enum = sm.cur_state_enum

            if cur_state_enum == StateEnum.AT_ST1:
                gen_state = "ST1_AUDIO"
                real_text_id = -5000000
            elif cur_state_enum == StateEnum.CONT_ST0_FORCE_END:
                gen_state = "ST0_END"
                real_text_id = -5000002
            else:
                gen_state = "ST0_AUDIO_TEXT"
            
            req.aux_output_infos["gen_state"].append(gen_state)
            
            sm_input = StateMachineInput(text_id=real_text_id, semantic_id=semantic_id)
            trans = sm.process(sm_input)
            if trans and self.attn_tp_rank == 0:
                logger.info(f"\033[34m[[{req_idx=}, Rid {req.rid}] {sm.to_string()}]\033[0m")

            if sm.get_state() == StateEnum.ABORT:
                req.finished_reason = FINISH_ABORT(message=f"{sm.to_string()}")
                finish_rids.append(req.rid)

            if sm.get_state() == StateEnum.AT_ST0 and sm.get_step() == 1:
                # step1 比较特殊，我们需要抛弃 step0 当中无效的 Audio Codecs，但是需要保留 semantic id
                audio_ids[req_idx][1:].fill_(-1000013)

            req.aux_output_infos["audio_codes"].append(audio_ids[req_idx].tolist())
            req.aux_output_infos["text_codes"].append(real_text_id)
            text_ids[req_idx] = real_text_id
            output_text_ids[req_idx] = get_spt().CHAT_TEXT_FINALLY_END if real_text_id < 0 else real_text_id

        for req_idx in range(len(forward_batch.reqs)):
            process_req(req_idx)

        # forward_batch.temp_multi_ids.copy_(audio_ids)
        # next_token_ids 非空会跳过 FluentLLM 本身负责的采样
        # forward_batch.next_token_ids = text_ids

        output_embeddings = torch.zeros_like(text_logits_output.hidden_states)
        if len(decode_idx):
            output_embeddings_part = self.input_process_forward_decode(
                text_ids[decode_idx].clone(),
                audio_ids[decode_idx].clone(),
                batch_size=len(decode_idx),
                reqs=[forward_batch.reqs[i] for i in decode_idx],
            )
            output_embeddings[decode_idx] = output_embeddings_part
        
        for rid in finish_rids:
            del self.sm_dict[rid]

        return output_text_ids, audio_ids, output_embeddings

    def audio_forward(
        self,
        forward_batch: ForwardBatch,
        hidden_states: Tensor,
    ):

        # 1. 预计算每个 req 的 past_id 和 output_len
        output_lens = torch.tensor([len(req.aux_output_infos["text_codes"]) for req in forward_batch.reqs], dtype=torch.long)
        past_ids = torch.clamp(output_lens - self.audio_rep_penalty_window, min=0)

        # 2. 提取所有 past_audio_ids 并记录原始长度
        past_audio_ids_list = []
        for i, req in enumerate(forward_batch.reqs):
            past_audio_ids = req.aux_output_infos["audio_codes"][past_ids[i] : output_lens[i]]
            # print(f'req.text_codes:{req.aux_output_infos["text_codes"]}, req.aux_output_infos:{req.aux_output_infos}, output_lens[i]:{output_lens[i]}, past_ids[i]:{past_ids[i]}, past_audio_ids:{past_audio_ids}')
            past_audio_ids = torch.as_tensor(past_audio_ids).reshape(-1, self.audio_head_num)
            past_audio_ids_list.append(past_audio_ids)

        # 3. 使用 pad_sequence 批量填充 (向量化操作)
        padded_past_audio_ids = pad_sequence(
            past_audio_ids_list, batch_first=True, padding_value=-1  # 填充值
        )  # [batch_size, max_seq_len, audio_head_num]

        # 4. 调整维度顺序: [batch_size, audio_head_num, max_seq_len]
        padded_past_audio_ids = padded_past_audio_ids.transpose(1, 2).cuda()

        # 5. 获取 audio_repetition_penalty (每个 req 不同) -> (batch_size,)
        audio_repetition_penalty = torch.tensor(
            [self.audio_repetition_penalty for req in forward_batch.reqs], #  req.input_extra_infos[0]["audio_repetition_penalty"]
            dtype=torch.float32,
            device=hidden_states.device,  # 与 hidden_states 同设备
        )  # (batch_size,)
        repetition_penalty_type = [RepetitionPenaltyType.EXPONENTIAL]*len(forward_batch.reqs) # 默认用类乘
        # for i, req in enumerate(forward_batch.reqs):
        #     if "repetition_penalty_type" in req.input_extra_infos[0]:
        #         repetition_penalty_type[i] = req.input_extra_infos[0]["repetition_penalty_type"]

        # print(f"xxxxx audio_output_layer:{hidden_states.shape} {padded_past_audio_ids.shape} {audio_repetition_penalty.shape} {repetition_penalty_type}")
        # 6. 调用 forward (支持 bs > 1 和每个样本不同的 penalty)
        raw_logits_list, panelized_logits_list, audio_ids_list = self.audio_output_layer.forward(
            lm_output=hidden_states,  # (batch_size, 1, dim)
            past_audio_ids=padded_past_audio_ids,  # (batch_size, audio_head_num, max_seq_len)
            audio_repetition_penalty=audio_repetition_penalty,  # (batch_size,)
            repetition_penalty_type=repetition_penalty_type,
        )

        return raw_logits_list, panelized_logits_list, audio_ids_list

    def sample_inner(self, next_token_logits, past_text_ids, top_p, top_k, temperature, repetition_penalty):
        batch_size, vocab_size = next_token_logits.shape
        # 填充text_ids到长度一致
        max_length = max(len(row) for row in past_text_ids)
        padded_past_text_ids = [row + [-1] * (max_length - len(row)) for row in past_text_ids]
        past_output_text_ids = torch.tensor(padded_past_text_ids,device=next_token_logits.device)
        # 全转成tensor
        top_k = torch.tensor(top_k,device=next_token_logits.device)
        top_p = torch.tensor(top_p,device=next_token_logits.device)
        temperature = torch.tensor(temperature,device=next_token_logits.device)
        repetition_penalty = torch.tensor(repetition_penalty,device=next_token_logits.device)
        # 如果是贪婪直接返回
        if torch.all(top_k == 1):
            batch_next_token_ids = torch.argmax(next_token_logits, -1)
        else:
            # 保护
            top_k = torch.where(top_k < 1, torch.tensor(float(vocab_size), device=next_token_logits.device), top_k)
            top_p = torch.where((top_p <= 0) | (top_p > 1), torch.tensor(1.0, device=next_token_logits.device), top_p)
            safe_temperature = torch.where(temperature == 0.0, torch.ones_like(temperature, device=next_token_logits.device), temperature)

            # 重复惩罚
            penalize_logits,_ = self.audio_output_layer.penalize_repetition(
                past_output_text_ids,
                next_token_logits.unsqueeze(1), # 复用audio的重复惩罚，需要在中间插入一个维度，(batch_size,1,vocab_size)
                repetition_penalty,
                RepetitionPenaltyType.EXPONENTIAL,  # 使用累乘惩罚
            )
            penalize_logits = penalize_logits.squeeze(1) # 变回(batch_size,vocab_size)
            # 温度
            penalize_logits.div_(safe_temperature.unsqueeze(-1))
            penalize_logits[:] = torch.softmax(penalize_logits, dim=-1)
            probs = penalize_logits
            del penalize_logits

            need_k_first = True
            if need_k_first:
                # 先k后p
                probs = top_k_renorm_prob(probs, top_k)
                batch_next_token_ids = top_p_sampling_from_probs(probs, top_p)
            else:
                # 联合采样
                batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs,
                        top_k,
                        top_p,
                        filter_apply_order="joint",
                        check_nan=False,
                    )
        
        return batch_next_token_ids

EntryClass = [LongcatFlashOmniForCausalLM]
