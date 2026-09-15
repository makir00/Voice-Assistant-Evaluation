import io
import wave

import numpy as np
import sounddevice as sd


class Speaker:
    """Play WAV audio through the system speakers."""

    def play(
        self,
        audio_data: bytes,
    ) -> None:
        """Play WAV audio data."""

        with wave.open(
            io.BytesIO(audio_data),
            "rb",
        ) as wav_file:
            sample_rate = (
                wav_file.getframerate()
            )
            channels = (
                wav_file.getnchannels()
            )
            sample_width = (
                wav_file.getsampwidth()
            )
            frames = wav_file.readframes(
                wav_file.getnframes()
            )

        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        elif sample_width == 1:
            dtype = np.uint8
        else:
            raise ValueError(
                "Unsupported WAV sample width: "
                f"{sample_width}"
            )

        audio = np.frombuffer(
            frames,
            dtype=dtype,
        )

        if channels > 1:
            audio = audio.reshape(
                -1,
                channels,
            )

        print(
            "Playing assistant response..."
        )

        sd.play(
            audio,
            sample_rate,
        )
        sd.wait()