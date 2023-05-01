from typing import Any

from bark import SAMPLE_RATE, generate_audio, preload_models

import base64
import io
from scipy.io.wavfile import write

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        preload_models()

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
