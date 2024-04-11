# V4
import os
import torch


def main():
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = './models/v4_cyrillic.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/cyr/v4_cyrillic.pt',
                                       local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    example_text = ('Чалахай иирнең! Уғаа тузалығ чоох! Хайди палаларны пос тіліне ӱгредерге? Полған на чуртағҷызы '
                    'нимес ахча кӱҷӱрлеріне тоғасхан.')
    sample_rate = 48000
    speaker = 'b_kjh'
    save_dir = './results'
    audio_path = f'{save_dir}/test.wav'
    audio_path = model.save_wav(text=example_text,
                                ssml_text=None,
                                audio_path=audio_path,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=True,
                                put_yo=True)


if __name__ == '__main__':
    main()
