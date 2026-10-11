"""Streaming Silero v6 ONNX inference on a timestamp-preserving 16 kHz audio grid."""

import numpy as np

from vn_av_data.common.runtime import require_file


def audio_blocks(path, origin_s):
    import av

    cursor, pending = 0, np.empty(0, np.float32)
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

        def frames():
            for frame in container.decode(audio=0):
                yield from resampler.resample(frame)
            yield from resampler.resample(None)

        for frame in frames():
            if frame.pts is None:
                raise ValueError("Audio frame without timestamp")
            start = round((float(frame.pts * frame.time_base) - origin_s) * 16000)
            samples = frame.to_ndarray().reshape(-1)
            gap = max(0, start - cursor)
            # Bound each allocation even when a source contains a large timestamp gap.
            while gap:
                n = min(gap, 16000)
                pending = np.concatenate([pending, np.zeros(n, np.float32)])
                cursor += n
                gap -= n
                while len(pending) >= 512:
                    yield pending[:512]
                    pending = pending[512:]
            skip = max(0, cursor - start)
            samples = samples[skip:]
            cursor += len(samples)
            pending = np.concatenate([pending, samples])
            while len(pending) >= 512:
                yield pending[:512]
                pending = pending[512:]
        if len(pending):
            yield np.pad(pending, (0, 512 - len(pending)))


def speech_regions(probabilities, duration, threshold=0.5, silence_s=0.3, pad_s=0.15):
    active = np.asarray(probabilities) >= threshold
    edges = np.diff(np.r_[False, active, False].astype(int))
    spans = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a, b = start * 0.032, min(duration, end * 0.032)
        if spans and a - spans[-1][1] <= silence_s:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    padded = []
    for a, b in spans:
        if b - a < 0.25:
            continue
        a, b = max(0, a - pad_s), min(duration, b + pad_s)
        if padded and a <= padded[-1][1]:
            padded[-1][1] = b
        else:
            padded.append([a, b])
    return padded


def detect_speech(path, model, origin_s, duration, silence_s=0.3):
    import onnxruntime as ort

    require_file(model, "Silero VAD; run relations-setup --only media")
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    state, context, probabilities = np.zeros((2, 1, 128), np.float32), np.zeros(64, np.float32), []
    for samples in audio_blocks(path, origin_s):
        value = np.concatenate([context, samples])[None]
        output, state = session.run(
            None, {"input": value, "state": state, "sr": np.array(16000, np.int64)}
        )
        probabilities.append(float(output.ravel()[0]))
        context = samples[-64:]
    return speech_regions(probabilities, duration, silence_s=silence_s)
