from typing import Any

from bark import SAMPLE_RATE, generate_audio
from bark.generation import load_model, load_codec_model

import base64
import io
from scipy.io.wavfile import write


def preload_models():
    text = load_model(model_type="text", use_gpu=True, use_small=False, force_reload=False)
    coarse = load_model(model_type="coarse", use_gpu=True, use_small=False, force_reload=False)
    fine = load_model(model_type="fine", use_gpu=True, use_small=False, force_reload=False)
    codec = load_codec_model(use_gpu=True, force_reload=False)
    return { "text": text, "coarse": coarse, "fine": fine, "codec": codec }


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None


    def load(self):
        # Load model here and assign to self._model.
        text = load_model(model_type="text", use_gpu=True, use_small=False, force_reload=False)
        coarse = load_model(model_type="coarse", use_gpu=True, use_small=False, force_reload=False)
        fine = load_model(model_type="fine", use_gpu=True, use_small=False, force_reload=False)
        codec = load_codec_model(use_gpu=True, force_reload=False)
        self._model = { "text": text, "coarse": coarse, "fine": fine, "codec": codec }


    def preprocess(self, model_input: Any) -> Any:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return model_input

    def postprocess(self, model_output: Any) -> Any:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return model_output

    def predict(self, model_input: Any) -> Any:
        text_prompt = model_input
        audio_array = generate_audio(text_prompt)
        return arr_to_b64(audio_array)


def arr_to_b64(arr):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, SAMPLE_RATE, arr)
    wav_bytes = byte_io.read()
    audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
    return audio_data
