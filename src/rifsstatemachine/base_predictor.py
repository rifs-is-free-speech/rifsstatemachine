"""Model used when nothing else is available."""

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """Base Predicter used when nothing else is available."""

    @abstractmethod
    def __init__(self):
        """Initialize the model."""
        ...

    @abstractmethod
    def predict(self, x):
        """Predicts the transcription of the audio

        Parameters
        ----------
        record : np.array
            Audio data
        Returns
        -------
        str

        """
        ...


class Predictor(BasePredictor):
    """Class that loads the model and predicts the transcription of the audio"""

    def __init__(self):
        """Loads the model and the processor"""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch

        self.model = Wav2Vec2ForCTC.from_pretrained(
            "Alvenir/wav2vec2-base-da-ft-nst"
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = Wav2Vec2Processor.from_pretrained(
            "Alvenir/wav2vec2-base-da-ft-nst"
        )
        self.model.eval()


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
        import torch
        import re

        data = torch.tensor(record)
        input_dict = self.processor(
            data,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
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
