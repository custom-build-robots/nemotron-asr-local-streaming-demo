#!/usr/bin/env python3
"""
nim_asr_gradio_app.py

Live speech recognition in the browser, acting as a thin streaming client
against a local NVIDIA NIM (Parakeet 1.1b RNNT Multilingual, mode=str).

The model runs as a microservice; this app captures the microphone, converts
the audio to mono / 16 kHz / 16-bit PCM and streams it chunk by chunk to the
local NIM gRPC endpoint via the Riva streaming API. The audio never leaves
the machine.

Author: Ingmar Stapel (ai-box.eu)
Date:   2026-06-14
Note:   This application was created with the assistance of AI.

Requirements:
  - A running Parakeet streaming NIM (mode=str), port 50051
  - pip install gradio nvidia-riva-client numpy scipy  (inside the venv)
"""

import queue
import threading
import traceback
from datetime import datetime
from math import gcd

import numpy as np
import gradio as gr
from scipy import signal

import riva.client

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NIM_SERVER = "localhost:50051"   # local NIM gRPC endpoint
USE_SSL = False                  # local: no TLS needed
LANGUAGE_CODE = "de-DE"          # confirmed via the NIM log
SAMPLE_RATE = 16000              # Riva expects mono, 16 kHz, 16-bit PCM
STREAM_EVERY = 0.25              # send one audio chunk every 0.25 s
DEBUG = False                    # set True to print streaming details to the terminal


def log(msg):
    if DEBUG:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_streaming_config(language_code):
    """Same settings as the working CLI test."""
    return riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=language_code,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        ),
        interim_results=True,
    )


def to_pcm16_16k(sample_rate, data):
    """Convert a Gradio (sr, np.ndarray) chunk to mono, 16 kHz, 16-bit PCM bytes.

    IMPORTANT: Gradio delivers the microphone audio already on the int16 scale
    (values up to ~32767), sometimes as a float dtype. It must therefore NOT
    be clipped to [-1, 1] - doing so would crush the signal into full-scale
    noise. We keep everything on the int16 scale, resample to 16 kHz and return
    PCM bytes.
    """
    arr = np.asarray(data)
    is_float = np.issubdtype(arr.dtype, np.floating)
    arr = arr.astype(np.float32)

    if arr.ndim == 2:                       # stereo -> mono
        arr = arr.mean(axis=1)

    if is_float:
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak <= 1.0:                     # normalized float [-1, 1]
            arr = arr * 32767.0
        # otherwise: float already on the int16 scale -> leave unchanged
    # (integer dtype is already on the int16 scale)

    if sample_rate != SAMPLE_RATE:
        g = gcd(int(sample_rate), SAMPLE_RATE)
        arr = signal.resample_poly(arr, SAMPLE_RATE // g, int(sample_rate) // g)

    arr = np.clip(arr, -32768.0, 32767.0)   # clamp to the valid int16 range
    return arr.astype(np.int16).tobytes()


class RivaNimStreamer:
    """Holds one running streaming session against the local NIM."""

    def __init__(self, server, language_code, use_ssl=False):
        log(f"New Riva session -> {server}, lang={language_code}")
        auth = riva.client.Auth(uri=server, use_ssl=use_ssl)
        self.asr_service = riva.client.ASRService(auth)
        self.language_code = language_code
        self.audio_queue = queue.Queue()
        self._lock = threading.Lock()
        self.finalized = ""
        self.interim = ""
        self.error = None
        self.chunk_count = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _audio_generator(self):
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                log("Audio generator: sentinel received, stream ends.")
                break
            yield chunk

    def _run(self):
        try:
            log("Riva streaming generator starting ...")
            cfg = build_streaming_config(self.language_code)
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=self._audio_generator(),
                streaming_config=cfg,
            )
            for response in responses:
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    final = result.is_final
                    log(f"Riva {'FINAL  ' if final else 'partial'}: {transcript!r}")
                    with self._lock:
                        if final:
                            self.finalized += transcript
                            self.interim = ""
                        else:
                            self.interim = transcript
            log("Riva generator finished.")
        except Exception:
            log("ERROR in Riva thread:\n" + traceback.format_exc())
            with self._lock:
                self.error = traceback.format_exc().strip().splitlines()[-1]

    def push(self, sample_rate, data):
        arr = np.asarray(data)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        pcm = to_pcm16_16k(sample_rate, data)
        self.chunk_count += 1
        log(f"Chunk #{self.chunk_count}: sr={sample_rate} shape={arr.shape} "
            f"dtype={arr.dtype} peak={peak:.1f} -> {len(pcm)} bytes")
        if pcm:
            self.audio_queue.put(pcm)

    def stop(self):
        self.audio_queue.put(None)

    def text(self):
        with self._lock:
            if self.error:
                return f"[Error] {self.error}"
            return (self.finalized + self.interim).strip()


# --------------------------------------------------------------------------
# Gradio callbacks
# --------------------------------------------------------------------------
def on_start(_state):
    log("=== Recording started ===")
    return RivaNimStreamer(NIM_SERVER, LANGUAGE_CODE, USE_SSL), ""


def on_stream(new_chunk, state):
    if state is None:
        log("on_stream: no state -> creating streamer "
            "(start_recording may not have fired)")
        state = RivaNimStreamer(NIM_SERVER, LANGUAGE_CODE, USE_SSL)
    if new_chunk is not None:
        sr, data = new_chunk
        state.push(sr, data)
    return state, state.text()


def on_stop(state):
    log("=== Recording stopped ===")
    if state is not None:
        state.stop()
        return state, state.text()
    return state, ""


def on_reset(state):
    """Clear the transcript and end the current session for a fresh start."""
    log("=== Reset ===")
    if state is not None:
        state.stop()
    return None, ""


with gr.Blocks(title="NVIDIA NIM - German Live Speech Recognition (local)") as demo:
    gr.Markdown(
        "## NVIDIA NIM - German Live Speech Recognition (local)\n"
        f"Streaming client against `{NIM_SERVER}` - your audio stays on the machine."
    )
    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        label="Microphone (speak German)",
    )
    transcript = gr.Textbox(label="Transcript (live)", lines=8, interactive=False)
    reset_btn = gr.Button("Reset")

    # gr.State holds the NIM session per browser session
    session = gr.State(value=None)

    mic.start_recording(on_start, [session], [session, transcript])
    mic.stream(on_stream, [mic, session], [session, transcript],
               stream_every=STREAM_EVERY)
    mic.stop_recording(on_stop, [session], [session, transcript])
    reset_btn.click(on_reset, [session], [session, transcript])


if __name__ == "__main__":
    # share=True creates a temporary public HTTPS link (handy because the
    # browser needs HTTPS for microphone access). Set to False for local-only.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
