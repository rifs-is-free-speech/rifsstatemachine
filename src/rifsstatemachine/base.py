"""Base classes for all statemachine related containers"""

import pendulum
import numpy as np

from abc import ABC, abstractmethod
from queue import Queue
from itertools import cycle

from rifsstatemachine.states import State


class StateMachine(ABC):
    """Base class for all StateMachines, which include
    the Recorder, Transcriber and Splitter.
    """

    _state = None
    _setup_ran = False

    @abstractmethod
    def __init__(self):
        """Initialize the StateMachine"""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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

    @abstractmethod
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
        ...

    def setup(self, verbose=False, quiet=False) -> None:
        """Setup the StateMachine

        Parameters
        ----------
        verbose : bool, optional
            If True, print out the state transitions, by default False
        quiet : bool, optional
            If True, do not print out any messages, by default False

        Returns
        -------
        None
        """
        self.verbose = verbose
        self.quiet = quiet
        self.audio_queue = Queue()
        self.start_time = pendulum.now()
        self.time = 0
        self.final_transcription = ""
        self.transcriptions_segments = []
        self.background_steps = 500
        self.samplerate = 16000
        self.noise_cutoff = 0.15
        self.queuebuffer = np.empty((1, 0))
        self.spinner = cycle(["-", "/", "|", "\\"])
        self._setup_ran = True

    def put(self, signal: np.array) -> None:
        """Puts the audio sample in the queue
        Parameters
        ----------
        signal : np.array
            The audio sample

        Returns
        -------
        None
        """
        audio_sample = signal.T
        self.queuebuffer = np.concatenate((self.queuebuffer, audio_sample), axis=1)
        if self.queuebuffer.shape[1] > 100:
            self.audio_queue.put(self.queuebuffer)
            self.queuebuffer = np.empty((1, 0))

    def setState(self, state: State) -> None:
        """Sets the state of the recorder
        Parameters
        ----------
        state : State
            The state of the recorder

        Returns
        -------
        None
        """
        if self.verbose and not self.quiet:
            print(f"Context: Transitioning to {type(state).__name__}")
        self._state = state
        self._state.context = self

    def process_next_audio_sample(self) -> None:
        """Processes the next audio sample in the queue

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.process(self.audio_queue.get())

    def process(self, signal: np.array) -> None:
        """Processes the audio sample

        Parameters
        ----------
        signal : np.array
            The audio sample

        Returns
        -------
        None
        """
        self.time += signal.shape[1]
        self._state.process(signal)
