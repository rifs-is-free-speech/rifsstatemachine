"""States for the rifs state machine."""

from __future__ import annotations

import pendulum
import numpy as np

from abc import ABC, abstractmethod
from rifsstatemachine.utils import calculate_rms


class State(ABC):
    """Abstract class for the states of the recorder finite state machine"""

    @property
    def context(self):
        """The context of the state machine"""
        return self._context

    @context.setter
    def context(self, context) -> None:
        """Sets the context of the state

        Parameters:
        -----------
        context : Context
            The context of the state

        Returns:
        --------
        None
        """
        self._context = context

    @abstractmethod
    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        ...


class Start(State):
    """The start state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        print(
            "Measuring background noises..."
        ) if self.context.verbose and not self.context.quiet else None
        self.context.bar = self.context.progressbar(self.context.background_steps)
        self.context.loading = next(self.context.bar)
        self.context.buffer = audio_sample
        self.context.background_step = 1
        self.context.loading()
        self.context.last_speech_buffer = np.empty((1, 0))
        self.context.last_speech_found_time = None
        self.context.last_predict_time = None
        self.context.speech_lenght = 0
        self.context.setState(MeasureBackgroundNoise())


class MeasureBackgroundNoise(State):
    """The measure background noise state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        if calculate_rms(audio_sample) > self.context.noise_cutoff:
            return
        self.context.buffer = np.concatenate(
            (self.context.buffer, audio_sample), axis=1
        )

        self.context.background_step += 1
        self.context.loading()
        if self.context.background_step >= self.context.background_steps:
            self.context.background_rms = calculate_rms(self.context.buffer)
            self.context.last_background_rms = pendulum.now()
            self.context.background_buffer = self.context.buffer
            self.context.buffer = np.empty((1, 0))
            try:
                next(self.context.bar)
            except StopIteration:
                pass
            self.context.setState(Wait())
            print(
                "Starting trascription...\n"
            ) if self.context.verbose and not self.context.quiet else None


class SkipMeasureBackgroundNoise(State):
    """The start state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        self.context.buffer = audio_sample
        self.context.last_background_rms = pendulum.now()
        self.context.background_rms = 0.01
        self.context.background_buffer = np.full(
            (1, 16000), self.context.background_rms
        )
        self.context.last_speech_buffer = np.empty((1, 0))
        self.context.last_speech_found_time = None
        self.context.last_predict_time = None
        self.context.speech_lenght = 0
        self.context.setState(Wait())


class Wait(State):
    """The wait state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        if self.context.buffer.shape[1] >= 7923:
            self.context.buffer = self.context.buffer[:, -7923:]

        self.context.buffer = np.concatenate(
            (self.context.buffer, audio_sample), axis=1
        )

        if calculate_rms(audio_sample) > self.context.background_rms * 2.5:
            self.context.listen_start = self.context.time
            self.context.last_predict = self.context.time
            self.context.last_check_time = self.context.time
            self.context.listen_buffer = np.empty((1, 0))
            self.context.setState(Aware())
            return

        self.context.background_buffer = np.concatenate(
            (self.context.background_buffer[:, -69361:], audio_sample), axis=1
        )
        if self.context.last_background_rms.add(seconds=1) < pendulum.now():
            self.context.background_rms = calculate_rms(self.context.background_buffer)
            self.context.last_background_rms = pendulum.now()
        if self.context.last_speech_found_time:
            seconds_since_last_speech_found = (
                self.context.time - self.context.last_speech_found_time
            ) / self.context.samplerate
            if seconds_since_last_speech_found > 3:
                if self.context.disable_background_noise:
                    if self.context.last_speech_buffer.shape[1] >= 100:
                        self.context.predict(self.context.last_speech_buffer)
                else:
                    self.context.predict(
                        np.concatenate(
                            (
                                self.context.background_buffer[:, :8000],
                                self.context.last_speech_buffer,
                                self.context.background_buffer[:, -8000:],
                            ),
                            axis=1,
                        )
                    )
                self.context.last_speech_buffer = np.empty((1, 0))
                self.context.last_predict_time = self.context.last_speech_found_time
                self.context.last_speech_found_time = None


class Aware(State):
    """The aware state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        self.context.buffer = np.concatenate(
            (self.context.buffer, audio_sample), axis=1
        )
        seconds_elapsed = (
            self.context.time - self.context.listen_start
        ) / self.context.samplerate
        last_check = (
            self.context.time - self.context.last_check_time
        ) / self.context.samplerate
        print(next(self.context.spinner), end="\r") if not self.context.quiet else None
        if seconds_elapsed <= 1:
            if last_check > 0.1:
                self.context.last_check_time = self.context.time
                if self.context.check_for_speech(self.context.buffer):
                    self.context.setState(Listen())
                    return
        else:
            self.context.background_buffer = np.concatenate(
                (self.context.background_buffer, self.context.buffer, audio_sample),
                axis=1,
            )
            self.context.setState(Wait())
            print("  ", end="\r")


class Listen(State):
    """The listen state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        self.context.buffer = np.concatenate(
            (self.context.buffer, audio_sample), axis=1
        )
        self.context.listen_for_speech(audio_sample)
        if self.context.listen_buffer.shape[1] >= 7923:
            self.context.listen_buffer = self.context.listen_buffer[:, -7923:]
        self.context.listen_buffer = np.concatenate(
            (self.context.listen_buffer, audio_sample), axis=1
        )

        seconds_elapsed = (
            self.context.time - self.context.listen_start
        ) / self.context.samplerate
        if self.context.disable_background_noise:
            if (
                seconds_elapsed > 2
                and calculate_rms(self.context.listen_buffer)
                < self.context.background_rms * 2.5
                or seconds_elapsed > 8
            ):
                self.context.setState(Impatient())
        else:
            if (
                calculate_rms(self.context.listen_buffer)
                < self.context.background_rms * 2.5
                or seconds_elapsed > 8
            ):
                self.context.setState(Impatient())


class Impatient(State):
    """The impatient state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample

        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """
        self.context.buffer = np.concatenate(
            (self.context.buffer, audio_sample), axis=1
        )

        self.context.listen_for_speech(audio_sample)
        self.context.listen_buffer = np.concatenate(
            (self.context.listen_buffer, audio_sample), axis=1
        )
        seconds_elapsed = (
            self.context.time - self.context.listen_start
        ) / self.context.samplerate

        if self.context.disable_background_noise:
            if seconds_elapsed < 3:
                self.context.setState(Listen())
                return

        if (seconds_elapsed < 3) and (
            calculate_rms(self.context.listen_buffer)
            < self.context.background_rms * 1.5
        ):
            self.context.setState(Predict())
        elif (seconds_elapsed < 6) and (
            calculate_rms(self.context.listen_buffer)
            < self.context.background_rms * 2.5
        ):
            self.context.setState(Predict())
        elif seconds_elapsed > 6:
            self.context.setState(Predict())
        else:
            if self.context.check_for_speech(
                np.concatenate(
                    (
                        self.context.background_buffer[:, :8000],
                        self.context.buffer,
                        self.context.background_buffer[:, -8000:],
                    ),
                    axis=1,
                )
            ):
                self.context.setState(Listen())
            else:
                self.context.buffer = np.empty((1, 0))
                self.context.setState(Wait())


class Predict(State):
    """The predict state of the recorder finite state machine"""

    def process(self, audio_sample: np.array) -> None:
        """Processes the audio sample


        Paramaters:
        -----------
        audio_sample: np.array
            The audio sample to process

        Returns:
        --------
        None
        """

        self.context.buffer = np.concatenate(
            (self.context.buffer, audio_sample), axis=1
        )

        seconds_elapsed = (
            self.context.time - self.context.listen_start
        ) / self.context.samplerate

        if self.context.last_speech_found_time:
            seconds_since_last_speech_found = (
                self.context.time - self.context.last_speech_found_time
            ) / self.context.samplerate
            if seconds_since_last_speech_found - seconds_elapsed > 3:
                if self.context.disable_background_noise:
                    if self.context.last_speech_buffer.shape[1] >= 100:
                        self.context.predict(self.context.last_speech_buffer)
                else:
                    self.context.predict(
                        np.concatenate(
                            (
                                self.context.background_buffer[:, :8000],
                                self.context.last_speech_buffer,
                                self.context.background_buffer[:, -8000:],
                            ),
                            axis=1,
                        )
                    )
                self.context.last_speech_buffer = np.empty((1, 0))
                self.context.last_predict_time = self.context.last_speech_found_time
                self.context.last_speech_found_time = None

        if self.context.last_predict_time:
            seconds_since_last_predict = (
                self.context.time - self.context.last_predict_time
            ) / self.context.samplerate
            if seconds_since_last_predict - seconds_elapsed > 3:
                self.context.final_transcription += "\n\n"
                self.context.last_predict_time = None
        if seconds_elapsed + self.context.speech_lenght < 3:
            if self.context.check_for_speech(self.context.buffer):
                self.context.last_speech_found_time = self.context.time
                self.context.last_speech_buffer = np.concatenate(
                    (self.context.last_speech_buffer, self.context.buffer), axis=1
                )
                if self.context.disable_background_noise:
                    self.context.halfway(
                        self.context.last_speech_buffer,
                    )
                else:
                    self.context.halfway(
                        np.concatenate(
                            (
                                self.context.last_speech_buffer,
                                self.context.background_buffer[:, -8000:],
                            ),
                            axis=1,
                        )
                    )

                self.context.speech_lenght += seconds_elapsed
            self.context.buffer = audio_sample
            self.context.setState(Wait())
        else:
            if self.context.disable_background_noise:
                self.context.predict(
                    np.concatenate(
                        (
                            self.context.last_speech_buffer,
                            self.context.buffer,
                        ),
                        axis=1,
                    )
                )
            else:
                self.context.predict(
                    np.concatenate(
                        (
                            self.context.background_buffer[:, :16000],
                            self.context.last_speech_buffer,
                            self.context.buffer,
                            self.context.background_buffer[:, -16000:],
                        ),
                        axis=1,
                    )
                )
            self.context.buffer = audio_sample
            self.context.last_speech_buffer = np.empty((1, 0))
            self.context.last_predict_time = self.context.time
            self.context.speech_lenght = 0
            self.context.setState(Wait())
