import os
import torch
import torchaudio
from flask import Flask, request

app = Flask(__name__)

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = './models/denoise_model_sns_latest.pt'
model = torch.jit.load(local_file)
torch._C._jit_set_profiling_mode(False)
model.to(device)


def read_audio(
        path: str,
        sampling_rate: int = 24000
):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=sampling_rate
        )
        wav = transform(wav)
        sr = sampling_rate
    assert sr == sampling_rate
    return wav * 0.95


def save_audio(
        path: str,
        tensor: torch.Tensor,
        sampling_rate: int = 48000
):
    torchaudio.save(path, tensor, sampling_rate)


def denoise_sample(
        audio_path: str,
        save_path: str,
        sampling_rate: int = 48000
):
    audio = read_audio(audio_path).to(device)
    out = model(audio).flatten().unsqueeze(0)
    save_audio(save_path, out.cpu(), sampling_rate)


for _ in range(10):
    audio_path1 = './results/test.wav'
    new_audio_path1 = audio_path1.replace('.wav', '_denoise.wav')
    denoise_sample(audio_path=audio_path1,
                   save_path=new_audio_path1,
                   sampling_rate=48000)


@app.route('/denoise')
def denoise():
    audio_path = request.args.get('text')
    new_audio_path = audio_path.replace('.wav', '_denoise.wav')
    denoise_sample(audio_path=audio_path,
                   save_path=new_audio_path,
                   sampling_rate=48000)

    return os.path.abspath(new_audio_path)


if __name__ == '__main__':
    app.run(port=13301)
