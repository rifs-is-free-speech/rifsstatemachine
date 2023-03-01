"""Recorder class for recording and splitting speech online"""

import os

import numpy as np
from collections import namedtuple
from math import log2, exp

from rifsstatemachine.states import Start
from rifsstatemachine.base import StateMachine


class Recorder(StateMachine):
    """Splitter used to split 1 wav file into multiple sound bits of speech."""

    def __init__(self, model, verbose: bool = False, quiet: bool = False, notebook: bool = False) -> None:
        """Initialize the StateMachine

        Parameters
        ----------
        model : Model
            Optional model to use to transcribe the signal.
        verbose : bool
            Whether to print the state transitions
        quiet : bool
            Whether to silence the output
        notebook : bool
            Whether to use the notebook version of the recorder

        """
        self.setup(verbose=verbose, quiet=quiet)
        self.model = model
        self.setState(Start())
        self.disable_background_noise = False
        self.notebook = notebook
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
        if signal.shape[1] >= 100:
            return True if self.model.predict(signal) != "" else False
        else:
            return False

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
        if self.buffer.shape[1] < 417:
            return
        seconds_elapsed = (self.time - self.listen_start) / self.samplerate
        seconds_since_last_prediction = (
            self.time - self.last_predict
        ) / self.samplerate
        if seconds_since_last_prediction > (log2(exp(seconds_elapsed))) * 0.2:
            self.last_predict = self.time
            transcription = self.model.predict(
                np.concatenate((self.last_speech_buffer, self.buffer), axis=1)
            )
            if not self.notebook:
                if len(transcription) + 2 > os.get_terminal_size().columns:
                    transcription = (
                        transcription[: os.get_terminal_size().columns - 5] + "..."
                    )
            else:
                transcription = (
                        transcription[: 80 - 5] + "..."
                )
            print(f"  {transcription}", end="\r") if not self.quiet else None

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
        print(f"Utterance: {utterance}") if self.verbose and not self.quiet else None
        self.utterance_segments.append(utterance)

        if transcription != "":
            if not self.notebook:
                print(
                    " " * os.get_terminal_size().columns, end="\r"
                ) if not self.quiet else None
            else:
                print(
                    " " * 80, end="\r"
                ) if not self.quiet else None
            print(f"{transcription}") if not self.quiet else None
            self.final_transcription += transcription + " "

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
        self.last_predict = self.time
        transcription = self.model.predict(signal)

        if not self.notebook:

            if len(transcription) + 2 > os.get_terminal_size().columns:
                transcription = transcription[: os.get_terminal_size().columns - 5] + "..."
            print(
                " " * os.get_terminal_size().columns, end="\r"
            ) if not self.quiet else None
            print(f"  {transcription}", end="\r") if not self.quiet else None
        else:
            if len(transcription) + 2 > 80:
                transcription = transcription[: 80 - 5] + "..."
            print(
                " " * 80, end="\r"
            ) if not self.quiet else None
            print(f"  {transcription}", end="\r") if not self.quiet else None