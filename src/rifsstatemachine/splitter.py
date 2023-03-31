"""Splitter used to split 1 wav file into multiple sound bits of speech."""

import numpy as np
from collections import namedtuple

from rifsstatemachine.states import SkipMeasureBackgroundNoise
from rifsstatemachine.base import StateMachine


class Splitter(StateMachine):
    """Splitter used to split 1 wav file into multiple sound bits of speech."""

    def __init__(self, model=None, verbose: bool = False, quiet: bool = False) -> None:
        """Initialize the StateMachine

        Parameters
        ----------
        model : Model
            Optional model to use to check for speech
        verbose : bool
            Whether to print the state transitions
        quiet : bool
            Whether to silence the output
        """
        self.setup(verbose=verbose, quiet=quiet)
        self.model = model
        self.setState(SkipMeasureBackgroundNoise())
        self.disable_background_noise = True
        self.utterance_tuple = namedtuple("Utterance", "start end transcription signal")

    def check_for_speech(self, signal: np.array) -> bool:
        """Checks if there is speech in the audio sample
        Parameters
        ----------
        signal : np.array
            The audio sample
        Returns
        -------
        bool
            True if there is speech, False if there is not
        """
        if self.model:
            if signal.shape[1] >= 100 :

                if signal.shape[1] >= 3 * self.samplerate:
                    signal = signal[:, -3 * self.samplerate :]

                return True if self.model.predict(signal) != "" else False
            else:
                return False
        else:
            return True

    def listen_for_speech(self, audio_sample: np.array) -> None:
        """Processes the audio sample. Used to print the audio sample for
        state StateMachines that are recording.

        Parameters
        ----------
        signal : np.array
            The audio sample

        Returns
        -------
        None
        """
        pass

    def predict(self, signal: np.array) -> None:
        """Predicts on the signal of the StateMachine

        Parameters
        ----------
        signal : np.array
            The audio sample

        Returns
        -------
        None
        """
        if self.model:
            transcription = self.model.predict(signal)
            print(
                f"Prediction: {transcription}"
            ) if self.verbose and not self.quiet else None
            utterance = self.utterance_tuple(
                start=self.listen_start,
                end=self.time,
                transcription=transcription,
                signal=signal,
            )
            print(
                f"Utterance: {utterance}"
            ) if self.verbose and not self.quiet else None
            self.utterance_segments.append(utterance)
        else:
            utterance = self.utterance_tuple(
                start=self.listen_start,
                end=self.time,
                transcription="",
                signal=signal,
            )
            print(
                f"Utterance: {utterance}"
            ) if self.verbose and not self.quiet else None
            self.utterance_segments.append(utterance)

    def halfway(self, signal: np.array) -> None:
        """Prints the transcription halfway through the recording

        Parameters
        ----------
        signal : np.array
            The audio sample

        Returns
        -------
        None
        """
        print(
            "Halfway is skipped for this StateMachine"
        ) if self.verbose and not self.quiet else None
