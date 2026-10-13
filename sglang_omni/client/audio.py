# SPDX-License-Identifier: Apache-2.0
"""Audio encoding utilities.

Converts raw audio data (numpy arrays, torch tensors, or raw bytes) into
various output formats (WAV, MP3, FLAC, etc.) for API responses.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import math
import shutil
import struct
from functools import cache

import numpy as np
from scipy.signal import firwin

logger = logging.getLogger(__name__)

# Supported output formats and their MIME types
FORMAT_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

# Default sample rate for generated audio
DEFAULT_SAMPLE_RATE = 24000
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
RESAMPLER_OUTPUT_BLOCK_SAMPLES = 4096

# Configurations for PyAV encoding
PYAV_ENCODE_CONFIGS = {
    "opus": {
        "container": "ogg",
        "codecs": ("libopus", "opus"),
        "valid_rates": {8000, 12000, 16000, 24000, 48000},
    },
    "aac": {
        "container": "adts",
        "codecs": ("aac",),
        "valid_rates": {
            7350,
            8000,
            11025,
            12000,
            16000,
            22050,
            24000,
            32000,
            44100,
            48000,
            64000,
            88200,
            96000,
        },
    },
    "mp3": {
        "container": "mp3",
        "codecs": ("libmp3lame", "mp3"),
        "valid_rates": {
            8000,
            11025,
            12000,
            16000,
            22050,
            24000,
            32000,
            44100,
            48000,
            64000,
            88200,
            96000,
        },
    },
}


def audio_sample_rate_unavailable_reason(
    response_format: str, sample_rate: int
) -> str | None:
    """Return why an exact output sample rate is unsupported by an encoder."""

    config = PYAV_ENCODE_CONFIGS.get(response_format)
    if config is None:
        return None
    else:
        pass
    valid_rates = config["valid_rates"]
    if sample_rate in valid_rates:
        return None
    else:
        pass
    supported = ", ".join(str(rate) for rate in sorted(valid_rates))
    return (
        f"sample_rate={sample_rate} is not supported for "
        f"response_format={response_format!r}; supported rates: {supported}"
    )


class StreamingAudioResampler:
    """Stateful rational FIR resampler with chunk-boundary invariant output."""

    def __init__(self, source_sample_rate: int, target_sample_rate: int) -> None:
        if source_sample_rate <= 0 or target_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        else:
            pass
        divisor = math.gcd(source_sample_rate, target_sample_rate)
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.up_factor = target_sample_rate // divisor
        self.down_factor = source_sample_rate // divisor
        self.identity = self.up_factor == self.down_factor
        self.total_input_samples = 0
        self.next_output_index = 0
        self.finished = False
        self.channel_shape: tuple[int, ...] | None = None
        self.buffer = np.empty((0,), dtype=np.float32)
        self.buffer_start = 0

        if self.identity:
            self.filter_coefficients_by_phase = np.empty((0, 0), dtype=np.float64)
            self.filter_tap_offsets = np.empty((0,), dtype=np.int64)
            self.delay_output_samples = 0
            self.maximum_phase_taps = 0
        else:
            maximum_factor = max(self.up_factor, self.down_factor)
            half_length = 10 * maximum_factor
            half_length = math.ceil(half_length / self.down_factor) * self.down_factor
            coefficients = firwin(
                2 * half_length + 1,
                1.0 / maximum_factor,
                window=("kaiser", 5.0),
            ).astype(np.float64)
            coefficients *= self.up_factor
            filter_phases = tuple(
                coefficients[phase :: self.up_factor] for phase in range(self.up_factor)
            )
            self.delay_output_samples = half_length // self.down_factor
            self.maximum_phase_taps = max(
                len(phase_coefficients) for phase_coefficients in filter_phases
            )
            self.filter_coefficients_by_phase = np.zeros(
                (self.up_factor, self.maximum_phase_taps), dtype=np.float64
            )
            for phase_index, phase_coefficients in enumerate(filter_phases):
                self.filter_coefficients_by_phase[
                    phase_index, : len(phase_coefficients)
                ] = phase_coefficients
            self.filter_tap_offsets = np.arange(self.maximum_phase_taps, dtype=np.int64)

    def push(self, audio: object) -> np.ndarray:
        """Append one source-rate chunk and return newly available output."""

        if self.finished:
            raise RuntimeError("cannot push audio after finishing the resampler")
        else:
            pass
        chunk = to_numpy(audio)
        if chunk.ndim == 0:
            chunk = chunk.reshape(1)
        else:
            pass
        channel_shape = chunk.shape[:-1]
        if self.channel_shape is None:
            self.channel_shape = channel_shape
            self.buffer = np.empty(channel_shape + (0,), dtype=np.float32)
        elif channel_shape != self.channel_shape:
            raise ValueError(
                "audio channel shape changed from "
                f"{self.channel_shape} to {channel_shape}"
            )
        else:
            pass
        if chunk.shape[-1] == 0:
            return self.empty_output()
        else:
            pass
        if self.identity:
            self.total_input_samples += int(chunk.shape[-1])
            self.next_output_index += int(chunk.shape[-1])
            return chunk.astype(np.float32, copy=False)
        else:
            pass
        self.buffer = np.concatenate((self.buffer, chunk), axis=-1)
        self.total_input_samples += int(chunk.shape[-1])
        output = self.produce_available(final=False)
        self.prune_buffer()
        return output

    def finish(self) -> np.ndarray:
        """Flush filter delay and return the final target-rate samples."""

        if self.finished:
            return self.empty_output()
        else:
            pass
        self.finished = True
        if self.identity:
            return self.empty_output()
        else:
            pass
        output = self.produce_available(final=True)
        self.buffer = self.empty_output()
        self.buffer_start = self.total_input_samples
        return output

    def produce_available(self, *, final: bool) -> np.ndarray:
        target_count = math.ceil(
            self.total_input_samples * self.up_factor / self.down_factor
        )
        output_blocks: list[np.ndarray] = []
        while self.next_output_index < target_count:
            block_end = min(
                self.next_output_index + RESAMPLER_OUTPUT_BLOCK_SAMPLES,
                target_count,
            )
            output_indices = np.arange(
                self.next_output_index, block_end, dtype=np.int64
            )
            raw_output_indices = self.delay_output_samples + output_indices
            high_rate_indices = raw_output_indices * self.down_factor
            source_indices = high_rate_indices // self.up_factor
            phase_indices = high_rate_indices % self.up_factor
            requested_block_samples = len(output_indices)
            if final:
                available_block_samples = requested_block_samples
            else:
                available_block_samples = int(
                    np.searchsorted(
                        source_indices,
                        self.total_input_samples,
                        side="left",
                    )
                )
            if available_block_samples == 0:
                break
            else:
                pass
            source_indices = source_indices[:available_block_samples]
            phase_indices = phase_indices[:available_block_samples]
            buffer_indices = (
                source_indices[:, None]
                - self.filter_tap_offsets[None, :]
                - self.buffer_start
            )
            valid_buffer_indices = (buffer_indices >= 0) & (
                buffer_indices < self.buffer.shape[-1]
            )
            bounded_buffer_indices = np.clip(
                buffer_indices,
                0,
                self.buffer.shape[-1] - 1,
            )
            input_samples = np.take(
                self.buffer,
                bounded_buffer_indices,
                axis=-1,
            )
            block_coefficients = self.filter_coefficients_by_phase[phase_indices]
            block_coefficients = np.where(
                valid_buffer_indices,
                block_coefficients,
                0.0,
            )
            coefficient_shape = (1,) * len(
                self.channel_shape or ()
            ) + block_coefficients.shape
            output_blocks.append(
                np.sum(
                    input_samples * block_coefficients.reshape(coefficient_shape),
                    axis=-1,
                    dtype=np.float64,
                ).astype(np.float32)
            )
            self.next_output_index += available_block_samples
            if available_block_samples < requested_block_samples:
                break
            else:
                pass
        if not output_blocks:
            return self.empty_output()
        else:
            pass
        return np.concatenate(output_blocks, axis=-1)

    def prune_buffer(self) -> None:
        raw_output_index = self.delay_output_samples + self.next_output_index
        source_index = raw_output_index * self.down_factor // self.up_factor
        oldest_required = max(0, source_index - self.maximum_phase_taps + 1)
        remove_count = min(
            max(oldest_required - self.buffer_start, 0), self.buffer.shape[-1]
        )
        if remove_count > 0:
            self.buffer = self.buffer[..., remove_count:]
            self.buffer_start += remove_count
        else:
            pass

    def empty_output(self) -> np.ndarray:
        return np.empty((self.channel_shape or ()) + (0,), dtype=np.float32)


def resample_audio(
    audio: object, source_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    """Resample complete audio with the same FIR used by streaming responses."""

    resampler = StreamingAudioResampler(source_sample_rate, target_sample_rate)
    first = resampler.push(audio)
    final = resampler.finish()
    if first.shape[-1] == 0:
        return final
    elif final.shape[-1] == 0:
        return first
    else:
        return np.concatenate((first, final), axis=-1)


@cache
def audio_encoding_unavailable_reason(response_format: str) -> str | None:
    """Return why the requested response format cannot be encoded."""

    if response_format == "flac":
        if importlib.util.find_spec("soundfile") is None:
            return "soundfile is required for response_format='flac'"
        else:
            pass
        return None
    else:
        pass

    if response_format not in PYAV_ENCODE_CONFIGS:
        return None
    else:
        pass

    if importlib.util.find_spec("av") is not None:
        return None
    else:
        pass

    if importlib.util.find_spec("pydub") is None:
        return "PyAV or pydub is required for " f"response_format={response_format!r}"
    else:
        pass

    if shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
        return (
            "PyAV, ffmpeg, or avconv is required for "
            f"response_format={response_format!r}"
        )
    else:
        pass

    return None


def to_numpy(
    audio: object,
) -> np.ndarray:
    """Convert audio data to a numpy float32 array.

    Accepts:
    - numpy ndarray
    - torch Tensor
    - list / tuple of numbers
    - bytes (assumed 16-bit PCM)
    """
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32, copy=False)
    else:
        pass

    # torch Tensor
    if hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        arr = audio.detach().cpu().float().numpy()
        return arr.astype(np.float32, copy=False)
    else:
        pass

    if isinstance(audio, (list, tuple)):
        return np.array(audio, dtype=np.float32)
    else:
        pass

    if isinstance(audio, bytes):
        # Assume 16-bit signed PCM
        arr = np.frombuffer(audio, dtype="<i2")
        return (arr.astype(np.float32) / 32768.0).astype(np.float32)
    else:
        pass

    raise TypeError(f"Unsupported audio type: {type(audio)}")


def apply_speed(
    audio: np.ndarray, speed: float, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Apply speed adjustment by resampling.

    Returns (adjusted_audio, adjusted_sample_rate).

    Raises:
        ValueError: If *speed* is zero or negative.
    """
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed}")
    else:
        pass
    if speed == 1.0:
        return audio, sample_rate
    else:
        pass

    # Speed up/slow down by changing the effective sample rate
    # Then resample to the original rate
    new_length = max(int(round(len(audio) / speed)), 1)
    old_idx = np.arange(len(audio), dtype=np.float64)
    new_idx = np.linspace(0.0, len(audio) - 1, num=new_length, dtype=np.float64)
    resampled = np.interp(new_idx, old_idx, audio).astype(np.float32)
    return resampled, sample_rate


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode audio as a WAV file (16-bit PCM)."""
    # Clamp to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    if pcm.ndim == 2:
        num_channels = int(pcm.shape[0])
        pcm_bytes = np.ascontiguousarray(pcm.T).tobytes()
    else:
        num_channels = 1
        pcm_bytes = pcm.tobytes()

    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(
        struct.pack(
            "<HHIIHH",
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
    )
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)

    return buf.getvalue()


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    else:
        pass
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    else:
        pass
    sample_count = audio.shape[-1]
    duration = sample_count / float(orig_sr)
    new_len = max(int(round(duration * target_sr)), 1)
    old_idx = np.arange(sample_count, dtype=np.float64)
    new_idx = np.linspace(0.0, sample_count - 1, num=new_len, dtype=np.float64)
    if audio.ndim == 1:
        return np.interp(new_idx, old_idx, audio).astype(np.float32)
    else:
        pass

    channels = audio.reshape(-1, sample_count)
    resampled = np.stack([np.interp(new_idx, old_idx, channel) for channel in channels])
    return resampled.reshape(audio.shape[:-1] + (new_len,)).astype(np.float32)


def encode_with_pyav(
    audio: np.ndarray,
    sample_rate: int,
    container_format: str,
    codecs: tuple[str, ...],
    valid_rates: set[int],
) -> bytes:
    import io

    import av

    if sample_rate not in valid_rates:
        nearest_rate = min(valid_rates, key=lambda r: abs(r - sample_rate))
        audio = resample_linear(audio, sample_rate, nearest_rate)
        sample_rate = nearest_rate
    else:
        pass

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_format)

    stream = None
    for codec in codecs:
        try:
            stream = container.add_stream(codec, rate=sample_rate)
            break
        except av.CodecError:
            continue

    if stream is None:
        container.close()
        codecs_str = "', '".join(codecs)
        raise RuntimeError(
            f"None of the codecs ('{codecs_str}') are supported by PyAV."
        )
    else:
        pass

    if audio.ndim == 1:
        layout = "mono"
        planar_audio = audio.reshape(1, -1)
    elif audio.ndim == 2 and audio.shape[0] in (1, 2):
        channels = int(audio.shape[0])
        layout = "mono" if channels == 1 else "stereo"
        planar_audio = audio
    else:
        raise ValueError(
            "PyAV encoding supports mono or stereo channel-first audio, "
            f"got shape {audio.shape}"
        )

    planar_audio = np.ascontiguousarray(planar_audio, dtype=np.float32)
    stream.layout = layout

    # FFmpeg expects float-planar (fltp) format for these codecs
    frame = av.AudioFrame.from_ndarray(planar_audio, format="fltp", layout=layout)
    frame.sample_rate = sample_rate

    for packet in stream.encode(frame):
        container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    return buf.getvalue()


def encode_pcm(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode audio as raw 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    if pcm.ndim == 2:
        pcm = np.ascontiguousarray(pcm.T)
    else:
        pass
    return pcm.tobytes()


def select_audio_delta(
    audio_data: object,
    *,
    emitted_samples: int,
    is_terminal: bool,
) -> tuple[np.ndarray | None, int]:
    """Return the un-emitted audio delta from a streaming audio chunk."""

    audio = to_numpy(audio_data)
    if audio.ndim > 1:
        audio = audio.squeeze()
    else:
        pass
    if audio.ndim > 1:
        # Streaming chunks are mono; downmix multi-channel payloads
        # (e.g. the 48 kHz stereo MOSS-TTS Local codec) instead of
        # silently dropping channels.
        channel_axis = 0 if audio.shape[0] < audio.shape[-1] else -1
        audio = audio.mean(axis=channel_axis).astype("float32")
    else:
        pass

    total_samples = int(audio.shape[-1]) if audio.ndim else 0
    if not is_terminal:
        return audio, emitted_samples + total_samples
    else:
        pass
    if total_samples <= emitted_samples:
        return None, emitted_samples
    else:
        pass
    return audio[emitted_samples:], total_samples


def encode_audio(
    audio: object,
    *,
    response_format: str = "wav",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_sample_rate: int | None = None,
    speed: float = 1.0,
    allow_format_fallback: bool = True,
) -> tuple[bytes, str]:
    """Encode audio data to the requested format.

    Args:
        audio: Raw audio data (numpy, torch tensor, list, bytes)
        response_format: Target format (wav, mp3, flac, opus, aac, pcm)
        sample_rate: Audio sample rate in Hz
        target_sample_rate: Optional output sample rate in Hz
        speed: Speed adjustment factor (1.0 = normal)
        allow_format_fallback: If True, return WAV when a compressed encoder is
            unavailable or the format is unknown. If False, raise ValueError.

    Returns:
        (encoded_bytes, mime_type)
    """
    fmt = response_format.lower().strip()
    if target_sample_rate is not None:
        if not MIN_SAMPLE_RATE <= target_sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError(
                f"target_sample_rate must be between {MIN_SAMPLE_RATE} "
                f"and {MAX_SAMPLE_RATE}"
            )
        else:
            pass
        unavailable_reason = audio_sample_rate_unavailable_reason(
            fmt, target_sample_rate
        )
        if unavailable_reason is not None:
            raise ValueError(unavailable_reason)
        else:
            pass
    else:
        pass
    arr = to_numpy(audio)

    if arr.ndim > 1:
        arr = arr.squeeze()
    else:
        pass
    if arr.ndim > 1:
        if arr.shape[0] > arr.shape[-1]:
            arr = arr.T
        else:
            pass
        # note(chenye): Preserve native stereo for supported non-WAV formats;
        # keep the historical mono behavior for raw PCM and unsupported
        # channel layouts.
        if fmt != "wav" and (
            fmt not in ("mp3", "flac", "opus", "aac") or arr.shape[0] != 2
        ):
            arr = arr.mean(axis=0).astype(np.float32)
        else:
            pass
    else:
        pass

    if speed != 1.0:
        if arr.ndim > 1:
            adjusted = [apply_speed(channel, speed, sample_rate) for channel in arr]
            arr = np.stack([channel for channel, _ in adjusted])
            sample_rate = adjusted[0][1]
        else:
            arr, sample_rate = apply_speed(arr, speed, sample_rate)
    else:
        pass

    if target_sample_rate is not None and target_sample_rate != sample_rate:
        arr = resample_audio(arr, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    else:
        pass

    mime = FORMAT_MIME_TYPES.get(fmt, "application/octet-stream")

    if fmt == "wav":
        return encode_wav(arr, sample_rate), mime
    else:
        pass

    if fmt == "pcm":
        return encode_pcm(arr, sample_rate), mime
    else:
        pass

    if fmt in ("opus", "aac", "mp3"):
        try:
            config = PYAV_ENCODE_CONFIGS[fmt]
            encoded_bytes = encode_with_pyav(
                arr,
                sample_rate,
                container_format=config["container"],
                codecs=config["codecs"],
                valid_rates=config["valid_rates"],
            )
            return encoded_bytes, mime
        except Exception as e:
            logger.warning(
                "PyAV %s encoding failed (%s); falling back to pydub/WAV",
                fmt.upper(),
                str(e),
            )

        try:
            from pydub import AudioSegment

            wav_bytes = encode_wav(arr, sample_rate)
            seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            buf = io.BytesIO()
            export_fmt = {"aac": "adts", "opus": "opus"}.get(fmt, fmt)
            seg.export(buf, format=export_fmt)
            return buf.getvalue(), mime
        except ImportError:
            if not allow_format_fallback:
                raise ValueError(f"pydub is required to encode response_format={fmt!r}")
            else:
                pass
            logger.warning(
                f"pydub not installed; falling back to WAV for {fmt} request"
            )
            return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]
        except Exception as exc:
            if not allow_format_fallback:
                raise ValueError(
                    f"Failed to encode response_format={fmt!r}: {exc}"
                ) from exc
            else:
                pass
            logger.warning(
                f"Failed to encode {fmt}; falling back to WAV", exc_info=True
            )
            return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]
    else:
        pass

    if fmt == "flac":
        try:
            import soundfile as sf

            buf = io.BytesIO()
            soundfile_audio = arr.T if arr.ndim == 2 else arr
            sf.write(
                buf,
                np.ascontiguousarray(soundfile_audio),
                sample_rate,
                format="FLAC",
            )
            return buf.getvalue(), mime
        except ImportError:
            if not allow_format_fallback:
                raise ValueError(
                    "soundfile is required to encode response_format='flac'"
                )
            else:
                pass
            logger.warning(
                "soundfile not installed; falling back to WAV for FLAC request"
            )
            return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]
    else:
        pass

    if not allow_format_fallback:
        raise ValueError(f"Unsupported audio format: {response_format!r}")
    else:
        pass
    logger.warning(f"Unknown audio format '{fmt}'; falling back to WAV")
    return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]


def audio_to_base64(
    audio: object,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_format: str = "wav",
) -> str:
    """Encode audio data to a base64 string.

    Useful for embedding audio in JSON responses (e.g. chat completions
    with audio modality).
    """
    audio_bytes, _ = encode_audio(
        audio, response_format=output_format, sample_rate=sample_rate
    )
    return base64.b64encode(audio_bytes).decode("ascii")
