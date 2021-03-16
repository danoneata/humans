# In order to use ESPnet do not forget to 
# $ source ~/src/espnet2/tools/venv/bin/activate

import os
import pdb
import sys

import soundfile

import torch

from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device


MODEL_PATH = "/home/doneata/src/espnet2/tools/venv/lib/python3.8/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e/exp/asr_train_asr_transformer_e18_raw_bpe_sp"
DEVICE = "cpu"
dtype = "float32"


def load_model():
    # Paths to pre-trained model
    asr_train_config = os.path.join(MODEL_PATH, "config.yaml")
    asr_model_file = os.path.join(MODEL_PATH, "54epoch.pth")

    asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, DEVICE
    )
    asr_model.to(dtype=getattr(torch, dtype)).eval()

    return asr_model


def encode(model, speech):
    speech = torch.tensor(speech)
    # data: (Nsamples, ) → (1, Nsamples)
    speech = speech.unsqueeze(0).to(getattr(torch, dtype))

    # lenghts: (1, )
    lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    batch = {"speech": speech, "speech_lengths": lengths}
    batch = to_device(batch, device=DEVICE)

    enc, _ = model.encode(**batch)
    return enc


def main():
    audio, _ = soundfile.read("data/grid/audio-16khz/s1/bbafsn.wav")
    model = load_model()
    enc = encode(model, audio)

    # from espnet2.bin.asr_inference import Speech2Text
    # speech2text = Speech2Text(os.path.join(MODEL_PATH, "config.yaml"), os.path.join(MODEL_PATH, "54epoch.pth"))
    # print(speech2text(audio))
    pdb.set_trace()

if __name__ == "__main__":
    main()
