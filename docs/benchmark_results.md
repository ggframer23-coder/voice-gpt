# Benchmark Results (audio/2026/01/260104_1806.mp3)

## Command

```bash
HF_HUB_OFFLINE=0 make benchmark AUDIO=audio/2026/01/260104_1806.mp3 MODEL_NAME=base.en RUN_WHISPERX=1 RUN_WHISPERX_DIARIZE=1
```

## Summary

| Engine | Real | User | Sys | Result |
| --- | --- | --- | --- | --- |
| whisper.cpp | 3.12 s | 10.33 s | 0.50 s | Success; transcript at `/tmp/stt-bench/whispercpp.txt`. |
| faster-whisper | 4.13 s | 6.25 s | 1.45 s | Success; transcript at `/tmp/stt-bench/faster-whisper.txt`. |
| parakeet | 3.67 s | 6.33 s | 1.49 s | Success (after installing `onnx-asr` via `uv pip install -e ".[parakeet]"`); transcript at `/tmp/stt-bench/parakeet.txt`. |
| whisperx-fp16 | 7.94 s | 6.58 s | 2.25 s | Failed: CPU/backend does not support float16 compute. |
| whisperx-fp32 | 12.04 s | 8.62 s | 4.38 s | Failed: PyTorch `weights_only=True` safe mode keeps blocking globals (collections.defaultdict, etc.) even after allowlisting `omegaconf.listconfig.ListConfig`, `omegaconf.base.ContainerMetadata`, `typing.Any`, and `list`. Additional allowlisting would be necessary to finish the float32 load. |
| whisperx-diarize | 8.89 s | 7.14 s | 2.72 s | Failed for the same float16 compute restriction that stops the non-diarized run. |

All transcripts (successful runs or placeholder files) live under `/tmp/stt-bench`. WhisperX cannot finish on this hardware: float16 compute is unsupported and the fallback to float32 hits PyTorch's safe serialization guardrails (`weights_only=True`) unless many globals are explicitly allowlisted.
