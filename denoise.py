import os
import torch
import torchaudio


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


def denoise(
        model: torch.nn.Module,
        audio_path: str,
        save_path: str = 'result.wav',
        device=torch.device('cpu'),
        sampling_rate: int = 48000
):
    audio = read_audio(audio_path).to(device)
    model.to(device)
    out = model(audio).flatten().unsqueeze(0)
    save_audio(save_path, out.cpu(), sampling_rate)


def main():
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = './models/denoise_model_sns_latest.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/denoise_models/sns_latest.jit',
                                       local_file)

    model = torch.jit.load(local_file)
    torch._C._jit_set_profiling_mode(False)
    torch.set_grad_enabled(False)
    model.to(device)

    denoise(model=model,
            audio_path='./results/test_500.wav',
            save_path='./results/test_denoise.wav',
            device=device,
            sampling_rate=48000)


if __name__ == "__main__":
    main()
