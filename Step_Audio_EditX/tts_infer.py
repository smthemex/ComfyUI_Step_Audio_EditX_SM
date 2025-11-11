import os
#import argparse
import torch

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from datetime import datetime
import torchaudio
import librosa
import soundfile as sf

# Project imports
from .tokenizer import StepAudioTokenizer
from .tts import StepAudioTTS
#from .model_loader import ModelSource
from .config.edit_config import get_supported_edit_types

logger = logging.getLogger(__name__)
# Save audio to temporary directory
def save_audio(filename, audio_data, sr, output_dir):
    """Save audio data to a temporary file with timestamp"""
    save_path = os.path.join(output_dir, f"{filename}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.info(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        pass 
        return "None"

    #return save_path


class StepAudioEditX:
    """Audio editing and voice cloning local inference class"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())

    def history_messages_to_show(self, messages):
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": raw_audio_part},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": edit_audio_part}
            ])
        return show_msgs

    def generate_clone(self,common_tts_engine, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state, filename_out):
        """Generate cloned audio"""
        logger.info("Starting voice cloning process")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        
    def generate_edit(self,common_tts_engine, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state, filename_out):
        """Generate edited audio"""
        logger.info("Starting audio editing process")

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                _, audio_save_path, previous_text = state["history_audio"][-1]
                audio_to_edit = audio_save_path
                text_to_use = previous_text
                logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                audio_save_path = save_audio(filename_out, audio_numpy, output_sr, self.args.output_dir)
                state["history_audio"].append((output_sr, audio_save_path, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }



def load_TTS_model(model_path,model_source="local",quantization="fp16",tts_model_id=None):
     # Initialize models
    try:
        # Load StepAudioTokenizer
        encoder = StepAudioTokenizer(
            os.path.join(model_path, "Step-Audio-Tokenizer"),
            model_source= model_source,
            funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")

        # Initialize common TTS engine directly
        common_tts_engine = StepAudioTTS(
            os.path.join(model_path, "Step-Audio-EditX"),
            encoder,
            model_source= model_source,
            tts_model_id=tts_model_id,
            quantization_config=None if quantization=="fp16" else quantization,
        )
        logger.info("✓ StepCommonAudioTTS loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        raise e
        
    return common_tts_engine



def infer_tts(common_tts_engine,args):
    #os.makedirs(args.output_dir, exist_ok=True)

    # Create StepAudioEditX instance
    step_audio_editx = StepAudioEditX(args)
    if args.edit_type == "clone":
        filename_out = os.path.basename(args.prompt_audio_path).split('.')[0] + "_cloned"
        _, state = step_audio_editx.generate_clone(common_tts_engine,
            args.prompt_text,
            args.prompt_audio_path,
            args.generated_text,
            args.edit_type,
            args.edit_info,
            step_audio_editx.init_state(),
            filename_out,
        )
        
    else:
        state = step_audio_editx.init_state()
        for iter_idx in range(args.n_edit_iter):
            logger.info(f"Starting edit iteration {iter_idx + 1}/{args.n_edit_iter}")
            filename_out = os.path.basename(args.prompt_audio_path).split('.')[0] + f"_edited_iter{iter_idx + 1}"   
            _, state = step_audio_editx.generate_edit(common_tts_engine,
                args.prompt_text,
                args.prompt_audio_path,
                args.generated_text,
                args.edit_type,
                args.edit_info,
                state,
                filename_out,
            )
    return  state

