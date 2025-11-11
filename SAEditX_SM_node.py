 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  SAEditX_SM_origin_dict
from .Step_Audio_EditX.tts_infer import load_TTS_model,infer_tts
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import io as ioi
import time
import torchaudio   
from omegaconf import OmegaConf
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_SAEditX_current_path = os.path.join(folder_paths.models_dir, "SAEditX")
if not os.path.exists(weigths_SAEditX_current_path):
    os.makedirs(weigths_SAEditX_current_path)

folder_paths.add_model_folder_path("SAEditX", weigths_SAEditX_current_path) # SAEditX dir


class Step_Audio_EditX_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Step_Audio_EditX_SM_Model",
            display_name="Step_Audio_EditX_SM_Model",
            category="Step_Audio_EditX",
            inputs=[
                io.Combo.Input("model_source",options= ["local", "auto", "modelscope", "huggingface"] ),
            ],
            outputs=[
                io.Custom("Step_Audio_EditX_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, model_source) -> io.NodeOutput:
        model = load_TTS_model(weigths_SAEditX_current_path,model_source)
        return io.NodeOutput(model)
    

class Step_Audio_EditX_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Step_Audio_EditX_SM_KSampler",
            display_name="Step_Audio_EditX_SM_KSampler",
            category="Step_Audio_EditX",
            inputs=[
                io.Custom("Step_Audio_EditX_SM_Model").Input("model"),
                io.Audio.Input("audio"),
                io.String.Input("prompt_text",default="这里输入音频的内容",multiline=True),
                io.String.Input("target_text",default="这里输入你要输出的声音",multiline=True),
                io.Combo.Input("task",options= ["clone","emotion","style","vad","denoise","paralinguistic","speed"]),
                io.String.Input("edit_info",default="sad",multiline=False),
                io.Boolean.Input("offload", default=True),
                io.Int.Input("n_edit_iter", default=2, min=1, max=20,step=1,display_mode=io.NumberDisplay.number),

            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="audio_path"),
            ],
        )
    @classmethod
    def execute(cls, model,audio,prompt_text,target_text,task,edit_info,offload,n_edit_iter) -> io.NodeOutput:    
        model.offload_cpu=offload

        # pre audio
        prompt_audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{time.strftime('%m%d%H%S')}_temp.wav")
        waveform=audio["waveform"].squeeze(0)
        buff = ioi.BytesIO()
        torchaudio.save(buff, waveform, audio["sample_rate"], format="wav")
        with open(prompt_audio_path, 'wb') as f:
            f.write(buff.getbuffer())

        args=OmegaConf.create(SAEditX_SM_origin_dict)
        args.prompt_audio_path=prompt_audio_path
        args.prompt_text=prompt_text.strip()
        args.generated_text=target_text.strip()
        args.edit_type=task
        args.edit_info=edit_info
        args.output_dir=folder_paths.get_output_directory()
        args.n_edit_iter=n_edit_iter

        # start infer
        print("start infer")
        state=infer_tts(model,args)
        cur_audio=state["history_messages"][-1]
        cur_info=state["history_audio"][-1]
        waveform=torch.from_numpy(cur_audio["edit_wave"][1]).cpu().float().unsqueeze(0) #torch.Size([1, 576000])
        sample_rate=cur_info[0] # 24000
        audio_path=cur_info[1]
        audio={"waveform": waveform.unsqueeze(0), "sample_rate":sample_rate }
        return io.NodeOutput(audio,audio_path)



from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/Step_Audio_EditX_SM_Extension")
async def get_hello(request):
    return web.json_response("Step_Audio_EditX_SM_Extension")

class Step_Audio_EditX_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Step_Audio_EditX_SM_Model,
            Step_Audio_EditX_SM_KSampler,
        ]
async def comfy_entrypoint() -> Step_Audio_EditX_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return Step_Audio_EditX_SM_Extension()



