from __future__ import annotations

import os
import queue
import itertools
import pendulum
import textwrap

import numpy as np

from alive_progress import alive_bar

from math import log2, exp

from rifsstatemachine.states import State, Start

rms = 0.0
db = -100.0


class Recorder:
    _state = None

    def __init__(self, model) -> None:
        """Sets the state of the recorder
        Parameters
        ----------
        model:
            The model to use for transcription
        """
        self.verbose = False
        self.quiet = False
        self.setState(Start())
        self.model = model
        self.fixer = None  # PunctFixer(language="da")
        self.audio_queue = queue.Queue()
        self.start_time = pendulum.now()
        self.time = 0
        self.final_transcription = ""
        self.background_steps = 500
        self.samplerate = 16000
        self.noise_cutoff = 0.15
        self.queuebuffer = np.empty((1, 0))
        self.fix_enable = False
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])

    def put(self, audio_sample: np.array) -> None:
        """Puts the audio sample in the queue
        Parameters
        ----------
        audio_sample : np.array
            The audio sample
        """
        audio_sample = audio_sample.T
        self.queuebuffer = np.concatenate((self.queuebuffer, audio_sample), axis=1)
        if self.queuebuffer.shape[1] > 100:
            self.audio_queue.put(self.queuebuffer)
            self.queuebuffer = np.empty((1, 0))

    def setState(self, state: State):
        """Sets the state of the recorder
        Parameters
        ----------
        state : State
            The state of the recorder
        """
        if self.verbose:
            print(f"Context: Transitioning to {type(state).__name__}")
        self._state = state
        self._state.context = self

    def process_next_audio_sample(self) -> None:
        """Processes the next audio sample in the queue"""
        self.process(self.audio_queue.get())

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample
        Parameters
        ----------
        audio_sample : np.array
            The audio sample
        """
        self.time += audio_sample.shape[1]
        self._state.process(audio_sample)

    def progressbar(self, length: int):

        """Create a progressbar"""
        with alive_bar(length) as bar:
            yield bar

    def check_for_speech(self, audio_sample: np.array) -> bool:
        """Checks if there is speech in the audio sample
        Parameters
        ----------
        audio_sample : np.array
            The audio sample
        Returns
        -------
        bool
            True if there is speech, False if there is not
        """

        if audio_sample.shape[1] < 417:
            return
        text = self.model.predict(audio_sample)
        if text != "":
            return True
        return False

    def listen_for_speech(self, audio_sample: np.array) -> None:
        """Predicts the transcription of the audio"""
        self.buffer = np.concatenate((self.buffer, audio_sample), axis=1)
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
            if len(transcription) + 2 > os.get_terminal_size().columns:
                transcription = (
                    transcription[: os.get_terminal_size().columns - 5] + "..."
                )
            print(f"  {transcription}", end="\r")

    def predict(self, audio_sample: np.array) -> None:
        """Predicts the transcription of the audio"""
        if audio_sample.shape[1] < 417:
            return
        transcription = self.model.predict(audio_sample)
        try:
            if self.fix_enable:
                transcription = self.fixer.punctuate(transcription)
        except AssertionError:
            pass
        except IndexError:
            pass
        except TypeError:
            pass
        if transcription != "":
            print(" " * os.get_terminal_size().columns, end="\r")
            print(f"{transcription}")
            self.final_transcription += transcription + " "

    def halfway(self, audio_sample: np.array) -> None:
        """Prints the transcription halfway through the recording"""
        if audio_sample.shape[1] < 417:
            return
        self.last_predict = self.time
        transcription = self.model.predict(audio_sample)
        try:
            if self.fix_enable:
                transcription = self.fixer.punctuate(transcription)
        except AssertionError:
            pass
        except IndexError:
            pass
        except TypeError:
            pass

        if len(transcription) + 2 > os.get_terminal_size().columns:
            transcription = transcription[: os.get_terminal_size().columns - 5] + "..."
        print(" " * os.get_terminal_size().columns, end="\r")
        print(f"  {transcription}", end="\r")


def record_and_predict(model) -> None:
    """Records audio and uses the model to predict the what was said
    Parameters:
    -----------
    model:
        The model to use for transcription

    Returns:
    --------
    None
    """
    import sounddevice as sd

    recorder = Recorder(model)

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
        print()
        print("Recording finished".center(os.get_terminal_size().columns))
        os.makedirs("recordings", exist_ok=True)
        path = os.path.abspath(os.path.curdir) + "/recordings"
        dt = pendulum.now().format("dddd DD MMMM YYYY HH:mm:ss")
        with open(path + f"/{dt}.txt", "w") as f:
            f.write(recorder.final_transcription)
        print(
            f"Final transcription saved to {path}".center(
                os.get_terminal_size().columns
            )
        )
        print()
        space = int(os.get_terminal_size().columns * 0.1)
        wrap_space = int(os.get_terminal_size().columns * 0.8)
        paragraphs = recorder.final_transcription.splitlines()
        textOut = "\n".join(
            [textwrap.fill(p, wrap_space, replace_whitespace=False) for p in paragraphs]
        )
        for line in textOut.splitlines():
            print(f"{' '*space}{line}")
        exit(0)
    except Exception as e:
        raise e
