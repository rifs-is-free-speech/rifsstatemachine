"""Functions used to use the state machine"""

import numpy as np

from more_itertools import grouper
from typing import Optional, NamedTuple
from collections.abc import Sequence
from rifsstatemachine.base_predictor import BasePredictor
from rifsstatemachine.splitter import Splitter
from rifsstatemachine.recorder import Recorder


def wav_to_utterances(
    signal: np.array, model: Optional[BasePredictor] = None
) -> Sequence[NamedTuple]:
    """Split the wav signal into utterance based on Splitter

    Paramaters:
    -----------
    signal: np.Array
        The full wav signal to split
    model: Optional[BasePredicter]
        The model to use to split the signal.
        If None will include segments without checking for speech.

    Returns:
    --------
    Sequence[NamedTuple]
        The wav signal split into utterances
    """
    splitter = Splitter(model, verbose=True)
    for chunk in grouper(signal, 10, incomplete="fill", fillvalue=0.0):
        splitter.process(np.asarray([chunk]))
    splitter.predict(
        np.concatenate(
            (
                splitter.background_buffer[:, :8000],
                splitter.last_speech_buffer,
                splitter.buffer,
                splitter.background_buffer[:, -8000:],
            ),
            axis=1,
        )
    )

    for utterance in splitter.utterance_segments:
        yield utterance


def wav_to_screen(signal: np.array, model: BasePredictor) -> Sequence[NamedTuple]:
    """Split the wav signal into utterances and print to screen based on Recorder

    Paramaters:
    -----------
    signal: np.Array
        The full wav signal to split
    model: BasePredicter
        The model to use to transcribe the signal.

    Returns:
    --------
    Sequence[NamedTuple]
        The wav signal split into utterances
    """
    splitter = Recorder(model, verbose=False)
    for chunk in grouper(signal, 10, incomplete="fill", fillvalue=0.0):
        splitter.process(np.asarray([chunk]))
    splitter.predict(
        np.concatenate(
            (
                splitter.background_buffer[:, :8000],
                splitter.last_speech_buffer,
                splitter.buffer,
                splitter.background_buffer[:, -8000:],
            ),
            axis=1,
        )
    )

    for utterance in splitter.utterance_segments:
        yield utterance


def record_to_screen(model: BasePredictor) -> Sequence[NamedTuple]:
    """Record speech signal into utterances and print to screen based on Recorder

    Paramaters:
    -----------
    model: BasePredicter
        The model to use to transcribe the signal.

    Returns:
    --------
    Sequence[NamedTuple]
        The wav signal split into utterances
    """
    import sounddevice as sd

    recorder = Recorder(model, verbose=False)

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        recorder.put(indata.copy().T)

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
                    recorder.background_buffer[:, :8000],
                    recorder.last_speech_buffer,
                    recorder.buffer,
                    recorder.background_buffer[:, -8000:],
                ),
                axis=1,
            )
        )
        for line in recorder.utterance_segments:
            print(line.transcription)
        for utterance in recorder.utterance_segments:
            yield utterance
    except Exception as e:
        raise e
