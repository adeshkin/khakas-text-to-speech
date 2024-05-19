import torch
import os
from flask import Flask, request
import uuid
from pydub import AudioSegment

app = Flask(__name__)

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = './models/v4_cyrillic.pt'
model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
torch._C._jit_set_profiling_mode(False)
torch.set_grad_enabled(False)
model.to(device)
sample_rate = 48000
speaker = 'b_kjh'

for i in range(10):
    text1 = 'Чалахай иирнең! Уғаа тузалығ чоох!'
    save_dir1 = './results'
    os.makedirs(save_dir1, exist_ok=True)
    audio_path1 = f'{save_dir1}/test_tts.wav'
    audio_path2 = model.save_wav(text=text1,
                                 ssml_text=None,
                                 audio_path=audio_path1,
                                 speaker=speaker,
                                 sample_rate=sample_rate,
                                 put_accent=True,
                                 put_yo=True)
    os.remove(audio_path2)


@app.route('/tts')
def tts():
    text = request.args.get('text')

    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    audio_path = f'{save_dir}/{uuid.uuid4()}.wav'

    audio_path = model.save_wav(text=text,
                                ssml_text=None,
                                audio_path=audio_path,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=True,
                                put_yo=True)

    audio_file = AudioSegment.from_file(audio_path, format="wav")
    # remove first 500 milliseconds due to noise
    new_audio_path = audio_path.replace('.wav', '_500.wav')
    audio_file[500:].export(new_audio_path, format="wav")
    os.remove(audio_path)

    return os.path.abspath(new_audio_path)


if __name__ == '__main__':
    app.run(port=13201)
