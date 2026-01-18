import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import stt.cli as cli


class TestEngineSelection(unittest.TestCase):
    def test_auto_engine_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_text("x", encoding="utf-8")
            with mock.patch("stt.cli._audio_duration_seconds", return_value=299.0):
                engine, duration = cli._resolve_engine_for_audio("auto", audio_path)
                self.assertEqual(engine, "faster-whisper")
                self.assertEqual(duration, 299.0)
            with mock.patch("stt.cli._audio_duration_seconds", return_value=300.0):
                engine, duration = cli._resolve_engine_for_audio("auto", audio_path)
                self.assertEqual(engine, "whisperx")
                self.assertEqual(duration, 300.0)

    def test_non_auto_engine_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_text("x", encoding="utf-8")
            engine, duration = cli._resolve_engine_for_audio("parakeet", audio_path)
            self.assertEqual(engine, "parakeet")
            self.assertIsNone(duration)


class TestTranscriptionMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(whisper_bin=Path("/bin/whisper"))

    def test_metadata_faster_whisper(self) -> None:
        metadata = cli._build_transcription_metadata(
            self.settings,
            "faster-whisper",
            "base.en",
            None,
            None,
            None,
            "medium",
            "cpu",
            False,
            None,
        )
        self.assertEqual(metadata["engine"], "faster-whisper")
        self.assertEqual(metadata["model"], "base.en")
        self.assertEqual(metadata["compute_type"], "int8")

    def test_metadata_whisperx(self) -> None:
        metadata = cli._build_transcription_metadata(
            self.settings,
            "whisperx",
            "base.en",
            None,
            None,
            None,
            "medium",
            "cpu",
            True,
            "float32",
        )
        self.assertEqual(metadata["engine"], "whisperx")
        self.assertEqual(metadata["model"], "medium")
        self.assertEqual(metadata["device"], "cpu")
        self.assertTrue(metadata["diarize"])
        self.assertEqual(metadata["compute_type"], "float32")

    def test_metadata_whispercpp(self) -> None:
        metadata = cli._build_transcription_metadata(
            self.settings,
            "whispercpp",
            "ggml-base.bin",
            None,
            None,
            None,
            "medium",
            "cpu",
            False,
            None,
        )
        self.assertEqual(metadata["engine"], "whispercpp")
        self.assertEqual(metadata["model"], "ggml-base.bin")
        self.assertEqual(metadata["whisper_bin"], "/bin/whisper")


class TestIterAudioFiles(unittest.TestCase):
    def test_skips_vad_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            audio = base / "audio.mp3"
            audio.write_text("x", encoding="utf-8")
            vad_dir = base / "sample.vad"
            vad_dir.mkdir()
            vad_audio = vad_dir / "clip.wav"
            vad_audio.write_text("x", encoding="utf-8")
            files = cli._iter_audio_files(base, recursive=True, extensions="mp3,wav")
            self.assertIn(audio, files)
            self.assertNotIn(vad_audio, files)


class TestAudioCommand(unittest.TestCase):
    def test_audio_defaults_to_audio_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                with mock.patch("stt.cli.ingest_dir") as ingest_mock:
                    cli.audio(
                        audio=None,
                        model="base.en",
                        save=None,
                        ingest=True,
                        source="whisper.cpp",
                        convert=True,
                        vad=False,
                        vad_dir=None,
                        vad_threshold=0.5,
                        vad_min_speech_ms=250,
                        vad_min_silence_ms=100,
                        vad_speech_pad_ms=30,
                        vad_max_speech_s=None,
                        vad_samples_overlap=0.1,
                        vad_bin=None,
                        vad_model=None,
                        vad_timestamps=True,
                        engine="auto",
                        parakeet_model=None,
                        parakeet_dir=None,
                        parakeet_quant=None,
                        whisperx_model="medium",
                        whisperx_device="cpu",
                        whisperx_diarize=False,
                        whisperx_diarize_model=None,
                        language="en",
                    )
                ingest_mock.assert_called_once()
                _, kwargs = ingest_mock.call_args
                self.assertEqual(kwargs["directory"], Path("audio"))
                self.assertTrue(kwargs["recursive"])
            finally:
                os.chdir(cwd)


class TestTranscribeToText(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            whisper_bin=None,
            offline=True,
            vad_bin=None,
            vad_model=None,
            parakeet_model="nemo-parakeet-tdt-0.6b-v3",
            parakeet_dir=None,
            parakeet_quant="int8",
        )

    def test_transcribe_to_text_uses_faster_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_text("x", encoding="utf-8")
            with mock.patch("stt.cli._resolve_engine_for_audio", return_value=("faster-whisper", 10.0)):
                with mock.patch(
                    "stt.transcribe.transcribe_audio_faster_whisper",
                    return_value=("hello", [{"word": "hello", "start": 0.0, "end": 1.0}]),
                ) as transcribe_mock:
                    with mock.patch("stt.cli.time.perf_counter", side_effect=[1.0, 2.5]):
                        text, metadata, words, _ = cli._transcribe_to_text(
                            self.settings,
                            audio_path,
                            "base.en",
                            convert=False,
                            vad=False,
                            vad_dir=None,
                            vad_threshold=0.5,
                            vad_min_speech_ms=250,
                            vad_min_silence_ms=100,
                            vad_speech_pad_ms=30,
                            vad_max_speech_s=None,
                            vad_samples_overlap=0.1,
                            vad_bin=None,
                            vad_model=None,
                            vad_timestamps=True,
                            engine="auto",
                            parakeet_model=None,
                            parakeet_dir=None,
                            parakeet_quant=None,
                            whisperx_model="medium",
                            whisperx_device="cpu",
                            whisperx_diarize=False,
                            whisperx_diarize_model=None,
                        )
                        transcribe_mock.assert_called_once()
                        self.assertEqual(text, "hello")
                        self.assertEqual(words[0]["word"], "hello")
                        self.assertEqual(metadata["transcription"]["engine"], "faster-whisper")
                        self.assertAlmostEqual(metadata["transcription"]["elapsed_seconds"], 1.5)

    def test_transcribe_to_text_enables_auto_diarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_text("x", encoding="utf-8")
        with mock.patch("stt.cli._resolve_engine_for_audio", return_value=("whisperx", 400.0)):
            with mock.patch("stt.cli._audio_duration_seconds", return_value=400.0):
                with mock.patch(
                    "stt.transcribe.transcribe_audio_whisperx",
                    return_value=("hello", [], {"whisperx": {"model": "medium"}}),
                ) as transcribe_mock:
                    cli._transcribe_to_text(
                        self.settings,
                        audio_path,
                        "base.en",
                        convert=False,
                        vad=False,
                        vad_dir=None,
                        vad_threshold=0.5,
                        vad_min_speech_ms=250,
                        vad_min_silence_ms=100,
                        vad_speech_pad_ms=30,
                        vad_max_speech_s=None,
                        vad_samples_overlap=0.1,
                        vad_bin=None,
                        vad_model=None,
                        vad_timestamps=True,
                        engine="auto",
                        parakeet_model=None,
                        parakeet_dir=None,
                        parakeet_quant=None,
                        whisperx_model="medium",
                        whisperx_device="cpu",
                        whisperx_diarize=False,
                        whisperx_diarize_model=None,
                    )
                    _, kwargs = transcribe_mock.call_args
                    self.assertTrue(kwargs["diarize"])


class TestMetadataPayload(unittest.TestCase):
    def test_select_transcription_only(self) -> None:
        metadata = {"transcription": {"engine": "whisperx"}, "extra": 1}
        payload = cli._select_metadata_payload(metadata, full=False)
        self.assertEqual(payload, {"engine": "whisperx"})

    def test_select_full(self) -> None:
        metadata = {"transcription": {"engine": "whisperx"}, "extra": 1}
        payload = cli._select_metadata_payload(metadata, full=True)
        self.assertEqual(payload, metadata)


class TestMetadataCommand(unittest.TestCase):
    def test_metadata_requires_filter_unless_all(self) -> None:
        with self.assertRaises(Exception):
            cli.metadata(entry_id=None, audio=None, full=False, all_entries=False)


if __name__ == "__main__":
    unittest.main()
