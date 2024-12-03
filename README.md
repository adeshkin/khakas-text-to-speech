### Description
#### Here you can find examples of the Flask application for speech synthesis in the [Khakas](https://en.wikipedia.org/wiki/Khakas_language) language.

### Installation

```commandline
pip install -r requirements.txt
```

### Usage

*Test tts model*
```commandline
python tts.py
```

*Test denoise model*
```commandline
python denoise.py
```

*Run flask app with tts model at localhost:13201:*
```commandline
python app_tts.py
```

*Run flask app with denoise model at localhost:13301*
```commandline
python app_denoise.py
```