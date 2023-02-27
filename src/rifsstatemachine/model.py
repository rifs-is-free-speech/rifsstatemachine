"""Model used when nothing else is available."""

import torch


class Predicter:
    """Class that loads the model and predicts the transcription of the audio"""

    def __init__(self):
        """Loads the model and the processor"""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.model = Wav2Vec2ForCTC.from_pretrained(
            "Alvenir/wav2vec2-base-da-ft-nst"
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = Wav2Vec2Processor.from_pretrained(
            "Alvenir/wav2vec2-base-da-ft-nst"
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, record):
        """Predicts the transcription of the audio
        Parameters
        ----------
        record : np.array
            Audio data
        Returns
        -------
        str
            Transcription of the audio
        """
        import re

        data = torch.tensor(record)
        input_dict = self.processor(
            data,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        logits = self.model(
            input_dict["input_values"]
            .squeeze(1)
            .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]

        regex = r"\[UNK\]unk\[UNK\]|\[UNK]"
        transcription = re.sub(regex, "", transcription)
        return transcription


if __name__ == "__main__":
    from rifsstatemachine.statemachine import Recorder
    import numpy as np
    import sounddevice as sd

    recorder = Recorder(Predicter())

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        recorder.put(indata.copy())

    print(f"Using sounddevice: {sd.query_devices(sd.default.device[0])['name']}")
    try:
        with sd.InputStream(
            samplerate=16000,
            device=sd.query_devices(sd.default.device[0])["name"],
            channels=1,
            callback=callback,
        ):
            while True:
                recorder.process_next_audio_sample()
    except KeyboardInterrupt:
        recorder.predict(
            np.concatenate(
                (
                    recorder.background_buffer[:, :16000],
                    recorder.last_speech_buffer,
                    recorder.buffer,
                    recorder.background_buffer[:, -16000:],
                ),
                axis=1,
            )
        )
        paragraphs = recorder.final_transcription.splitlines()
        for line in paragraphs:
            print(line)
        exit(0)
    except Exception as e:
        raise e
