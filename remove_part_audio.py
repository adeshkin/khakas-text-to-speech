from pydub import AudioSegment


def main():
    audio_file = AudioSegment.from_file('./results/test.wav', format="wav")
    # remove first 500 milliseconds
    audio_file[500:].export('./results/test_500.wav', format="wav")


if __name__ == "__main__":
    main()
