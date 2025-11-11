# ComfyUI_Step_Audio_EditX_SM
[ Step_Audio_EditX](https://github.com/stepfun-ai/Step-Audio-EditX)：the first open-source LLM-based audio model excelling at expressive and iterative audio editing—encompassing emotion, speaking style, and paralinguistics—alongside robust zero-shot text-to-speech (TTS) capabilities，try it in comfyUI

# Tips
* sampler菜单选择clone时为zero shot 语音克隆,上面的prompt文字内容跟输入音频一致,下面的是文生音频的目标prompt;
* 不选择clone时为edit模式,下方的prompt失效,按照工作流的note,在edit info输入tag来编辑你想要的style或者情绪;
* offload在Vram小于16时使用; 
* n edit iter 为编辑的轮次,一般2或者3就有好的效果;  
* When selecting 'clone' from the sampler menu, it is a'zero shot' voice clone. The prompt text 'above' is consistent with the input audio, and the prompt 'below' is for the text generated audio; 
* When 'clone' is not selected, it is in edit mode, and the prompt below becomes invalid. Follow the note in the workflow and enter the tag in 'edit_info' to edit the 'style' or 'emotion' you want; 
* 'Offload' is used when Vram is less than 16G;  
* 'n-edit-iter' is the round of editing, usually 2 or 3 has good results; 

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Step_Audio_EditX_SM

```

2.requirements  
----
* 精简掉 sox 和hyperpyyaml,diffuser版本因为tokens的问题,用diffuser==4.53.3版本,或者低于,否则无声;
* funasr库即便安装完成,注意控制台的信息,可能还需要装一个库(忘记是哪个了)
```
pip install -r requirements.txt
```

3.checkpoints 
----
* if offload （离线模式需要预下载模型）  
1.main [ Step-Audio-EditX ](https://huggingface.co/stepfun-ai/Step-Audio-EditX) or 魔搭  [ Step-Audio-EditX ](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX)  
2.tokens  [Step-Audio-Tokenizer ](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer)  or 魔搭 [Step-Audio-Tokenizer ](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer)  

```
├── ComfyUI/models/SAEditX
|     ├── Step-Audio-EditX
|          ├──all files #全部文件，包括子目录文件
|     ├── Step-Audio-Tokenizer
|          ├──all files #全部文件，包括子目录文件
```

# 4 Example
![](https://github.com/smthemex/ComfyUI_Step_Audio_EditX_SM/blob/main/example_workflows/example.png)

# 5 Usage Disclaimer
* 请勿用此方法或插件做任何违法行为,法网恢恢,疏而不漏,切勿以身试法!!
* 此插件只为开源开发者测试演示方法制作,未收取任何报酬,只是为爱发电;
* Do not use this model for any unauthorized activities, including but not limited to:
  * Voice cloning without permission
  * Identity impersonation
  * Fraud
Deepfakes or any other illegal purposes
* Ensure compliance with local laws and regulations, and adhere to ethical guidelines when using this model.
* The model developers are not responsible for any misuse or abuse of this technology.

# 6 Citation
```
@misc{yan2025stepaudioeditxtechnicalreport,
      title={Step-Audio-EditX Technical Report}, 
      author={Chao Yan and Boyong Wu and Peng Yang and Pengfei Tan and Guoqiang Hu and Yuxin Zhang and Xiangyu and Zhang and Fei Tian and Xuerui Yang and Xiangyu Zhang and Daxin Jiang and Gang Yu},
      year={2025},
      eprint={2511.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.03601}, 
}
```



