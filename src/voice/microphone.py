import sounddevice as sd
from scipy.io.wavfile import write


class Microphone:
    """Record audio from the system microphone."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def record(
        self,
        duration: int,
        output_path: str,
    ) -> str:
        """Record microphone input and save it as a WAV file."""

        print(
            "Recording..."
        )

        audio = sd.rec(
            int(
                duration * self.sample_rate
            ),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
        )

        sd.wait()

        write(
            output_path,
            self.sample_rate,
            audio,
        )

        print(
            f"Saved recording to: {output_path}"
        )

        return output_path