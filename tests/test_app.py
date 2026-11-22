import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import app


class VideoGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_output_dir = app.OUTPUT_DIR
        self.original_upload_dir = app.UPLOAD_DIR
        app.OUTPUT_DIR = Path(self.temp_dir.name) / "output"
        app.UPLOAD_DIR = Path(self.temp_dir.name) / "uploads"
        app.OUTPUT_DIR.mkdir()
        app.UPLOAD_DIR.mkdir()
        with app.jobs_lock:
            app.jobs.clear()
        with app.upload_sessions_lock:
            app.upload_sessions.clear()
        self.client = app.app.test_client()

    def tearDown(self):
        app.OUTPUT_DIR = self.original_output_dir
        app.UPLOAD_DIR = self.original_upload_dir
        self.temp_dir.cleanup()

    def test_homepage_lists_generation_modes_and_social_profiles(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Choose a workflow", response.data)
        for mode in ("transcript_only", "images", "video", "audio_images"):
            self.assertIn(f"/create?mode={mode}".encode(), response.data)
        self.assertIn(b'href="/settings"', response.data)

    def test_create_and_global_settings_pages(self):
        create = self.client.get("/create?mode=audio_images")
        transcript = self.client.get("/create?mode=transcript_only")
        settings = self.client.get("/settings")

        self.assertEqual(create.status_code, 200)
        self.assertIn(b'name="audio"', create.data)
        self.assertNotIn(b'name="transcript"', create.data)
        self.assertIn(b'Use open-model images (FLUX.1-schnell)', transcript.data)
        self.assertIn(b"local stick-figure scenes", transcript.data)
        self.assertNotIn(b"Character description", transcript.data)
        self.assertEqual(settings.status_code, 200)
        for setting in (b"platform", b"language", b"gender", b"tone", b"layout"):
            self.assertIn(setting, settings.data)

    def test_stylesheet_is_served_and_included_in_docker_image(self):
        response = self.client.get("/static/app.css")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b".mode-grid", response.data)
        response.close()
        dockerfile = Path(app.BASE_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY static/ static/", dockerfile)

    def test_processing_guide_is_available_from_the_app(self):
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Video Generator: flow, models, and libraries", response.data)
        for flow in (
            b"Flow 1: Transcript only",
            b"Flow 2: Transcript + images",
            b"Flow 3: Existing audio + images",
            b"Flow 4: Transcript + video",
        ):
            self.assertIn(flow, response.data)
        self.assertIn(b"Whisper is speech-to-text, not text-to-speech", response.data)
        response.close()

    def test_transcript_only_request_creates_pollable_job(self):
        with patch.object(app, "Thread") as thread_class:
            response = self.client.post(
                "/generate",
                data={
                    "transcript": "A short test transcript.",
                    "mode": "transcript_only",
                    "platform": "vertical",
                    "language": "hi",
                    "gender": "female",
                    "tone": "calm",
                    "layout": "fill",
                },
                headers={"Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 202)
        thread_class.assert_called_once()
        thread_class.return_value.start.assert_called_once()
        job_id = response.json["job_id"]
        status = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json["status"], "queued")
        self.assertIn("Shorts / Reels", status.json["platform"])
        self.assertEqual(status.json["voice"], app.VOICE_MODELS["hi_female"]["label"])
        self.assertEqual(status.json["tone"], "Calm / measured")
        self.assertIn("transcript-only stick-figure visuals", status.json["logs"][0]["message"])
        self.assertEqual(thread_class.call_args.kwargs["args"][4:7], ("hi_female", "calm", "fill"))
        self.assertEqual(status.json["visual_source"], "Local")
        self.assertEqual(thread_class.call_args.kwargs["args"][-1], "")

    def test_video_framing_preserves_image_by_default_and_supports_fill(self):
        fit_filter = app.frame_filter(720, 1280, "fit")
        fill_filter = app.frame_filter(720, 1280, "fill")

        self.assertIn("force_original_aspect_ratio=decrease", fit_filter)
        self.assertIn("pad=720:1280", fit_filter)
        self.assertNotIn("crop=", fit_filter)
        self.assertIn("force_original_aspect_ratio=increase", fill_filter)
        self.assertIn("crop=720:1280", fill_filter)

    def test_pillow_image_resize_preserves_fit_and_fill_framing(self):
        source = Path(self.temp_dir.name) / "wide.png"
        fit = Path(self.temp_dir.name) / "fit.jpg"
        fill = Path(self.temp_dir.name) / "fill.jpg"
        app.Image.new("RGB", (200, 100), "#ff0000").save(source)

        app.normalize_image(source, fit, 100, 100, "fit")
        app.normalize_image(source, fill, 100, 100, "fill")

        with app.Image.open(fit) as image:
            self.assertEqual(image.size, (100, 100))
            self.assertEqual(image.getpixel((50, 0)), (255, 255, 255))
            self.assertGreater(image.getpixel((50, 50))[0], 240)
        with app.Image.open(fill) as image:
            self.assertEqual(image.size, (100, 100))
            self.assertGreater(image.getpixel((0, 0))[0], 240)

    def test_transcript_only_draws_sequential_stickman_scenes(self):
        work_dir = Path(self.temp_dir.name) / "scenes"
        work_dir.mkdir()
        messages = []
        images = app.generate_stickman_images(
            "First, make a plan. Next, overcome the problem. Finally, celebrate success.",
            work_dir,
            320,
            180,
            lambda message, _progress: messages.append(message),
        )

        self.assertEqual(len(images), 3)
        self.assertTrue(all(path.is_file() for path in images))
        with app.Image.open(images[0]) as image:
            self.assertEqual(image.size, (320, 180))
        self.assertIn("Created stick-figure scene 3 of 3.", messages[-1])

    def test_transcript_only_creates_rich_animated_scenes(self):
        work_dir = Path(self.temp_dir.name) / "animated-scenes"
        work_dir.mkdir()
        messages = []
        animations = app.generate_stickman_animations(
            "First, develop an idea. Finally, celebrate success.",
            work_dir,
            640,
            360,
            4,
            lambda message, _progress: messages.append(message),
        )

        self.assertEqual(len(animations), 6)
        with app.Image.open(animations[0]) as animation:
            self.assertEqual(animation.size, (640, 360))
            self.assertGreater(animation.n_frames, 1)
        self.assertIn("Animated scene 6 of 6", messages[-1])

    def test_image_mode_requires_at_least_one_image(self):
        response = self.client.post(
            "/generate",
            data={
                "transcript": "A short test transcript.",
                "mode": "images",
                "platform": "youtube",
            },
            headers={"Accept": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload at least one image", response.json["error"])

    def test_audio_image_mode_requires_audio_and_images_without_a_transcript(self):
        response = self.client.post(
            "/generate",
            data={"mode": "audio_images", "platform": "youtube"},
            headers={"Accept": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload at least one image", response.json["error"])

        response = self.client.post(
            "/generate",
            data={
                "mode": "audio_images",
                "platform": "youtube",
                "images": (io.BytesIO(b"image-bytes"), "image.jpg"),
            },
            headers={"Accept": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload an audio file", response.json["error"])

    def test_audio_image_mode_starts_worker_with_chunked_audio(self):
        original_chunk_size = app.VIDEO_CHUNK_BYTES
        app.VIDEO_CHUNK_BYTES = 5
        try:
            start = self.client.post(
                "/uploads",
                json={"filename": "voice.mp3", "size": 8},
            )
            self.assertEqual(start.status_code, 201)
            upload_id = start.json["upload_id"]
            for index, chunk_body in enumerate((b"audio", b"123")):
                chunk = self.client.post(
                    f"/uploads/{upload_id}/chunks/{index}",
                    data=chunk_body,
                    content_type="application/octet-stream",
                )
                self.assertEqual(chunk.status_code, 200)

            with patch.object(app, "Thread") as thread_class:
                response = self.client.post(
                    "/generate",
                    data={
                        "mode": "audio_images",
                        "platform": "youtube",
                        "audio_upload_id": upload_id,
                        "images": (io.BytesIO(b"image-bytes"), "image.jpg"),
                    },
                    headers={"Accept": "application/json"},
                )

            self.assertEqual(response.status_code, 202)
            thread_args = thread_class.call_args.kwargs["args"]
            self.assertEqual(len(thread_args), 12)
            audio_path = thread_args[9]
            self.assertEqual(audio_path.suffix, ".mp3")
            self.assertEqual(audio_path.read_bytes(), b"audio123")
            self.assertNotIn(upload_id, app.upload_sessions)

            work_dir = thread_args[10]
            with patch.object(
                app,
                "create_assets",
                return_value=(
                    work_dir / "video.mp4",
                    work_dir / "slideshow.gif",
                    audio_path,
                ),
            ) as create_assets:
                app.generate_job(*thread_args)

            create_assets.assert_called_once()
            status = self.client.get(f"/jobs/{thread_args[0]}")
            self.assertEqual(status.json["status"], "complete")
        finally:
            app.VIDEO_CHUNK_BYTES = original_chunk_size

    def test_media_duration_falls_back_to_ffmpeg_without_ffprobe(self):
        ffmpeg_result = app.subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="Input #0, wav, from 'audio.wav':\n  Duration: 00:01:23.45, bitrate: 128 kb/s",
        )
        with (
            patch.object(app.shutil, "which", side_effect=[None, "/usr/bin/ffmpeg"]),
            patch.object(app.subprocess, "run", return_value=ffmpeg_result),
        ):
            duration = app.ffprobe_duration(Path("audio.wav"))

        self.assertEqual(duration, 83.45)

    def test_huggingface_opt_in_requires_token_and_consent_and_is_not_persisted(self):
        without_token = self.client.post(
            "/generate",
            data={
                "mode": "transcript_only",
                "transcript": "A short story.",
                "use_hf_images": "yes",
            },
            headers={"Accept": "application/json"},
        )
        self.assertEqual(without_token.status_code, 400)
        self.assertIn("Hugging Face token", without_token.json["error"])

        without_consent = self.client.post(
            "/generate",
            data={
                "mode": "transcript_only",
                "transcript": "A short story.",
                "use_hf_images": "yes",
                "hf_token": "hf-test-token",
            },
            headers={"Accept": "application/json"},
        )
        self.assertEqual(without_consent.status_code, 400)
        self.assertIn("inference notice", without_consent.json["error"])

        with patch.object(app, "Thread") as thread_class:
            response = self.client.post(
                "/generate",
                data={
                    "mode": "transcript_only",
                    "transcript": "A short story.",
                    "use_hf_images": "yes",
                    "hf_token": "hf-test-token",
                    "external_visual_consent": "yes",
                },
                headers={"Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 202)
        job_id = response.json["job_id"]
        self.assertEqual(thread_class.call_args.kwargs["args"][-1], "hf-test-token")
        self.assertEqual(app.jobs[job_id]["visual_source"], "Hugging Face")
        self.assertNotIn("hf-test-token", json.dumps(app.jobs[job_id]))

    def test_huggingface_scene_request_uses_flux_model_and_saves_image(self):
        client_options = {}
        request_data = {}

        class FakeInferenceClient:
            def __init__(self, **kwargs):
                client_options.update(kwargs)

            def text_to_image(self, prompt, **kwargs):
                request_data["prompt"] = prompt
                request_data["parameters"] = kwargs
                return app.Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")

        image_path = Path(self.temp_dir.name) / "generated.png"
        with patch.object(app, "InferenceClient", FakeInferenceClient):
            app.generate_huggingface_scene_image(
                "A character finds a lantern.",
                "The full story.",
                0,
                6,
                image_path,
                "hf-test-token",
                1024,
                576,
            )

        self.assertEqual(client_options["model"], "black-forest-labs/FLUX.1-schnell")
        self.assertEqual(client_options["provider"], "auto")
        self.assertEqual(client_options["token"], "hf-test-token")
        self.assertIn("scene 1 of 6", request_data["prompt"])
        self.assertEqual(request_data["parameters"]["num_inference_steps"], 4)
        with app.Image.open(image_path) as image:
            self.assertEqual(image.size, (1024, 576))

    def test_ai_scene_art_becomes_a_looping_animated_gif(self):
        source = Path(self.temp_dir.name) / "scene.png"
        animation = Path(self.temp_dir.name) / "scene.gif"
        app.Image.new("RGB", (800, 450), "#6084c8").save(source)

        app.create_animated_scene_gif(
            source,
            animation,
            "A character finds a lantern.",
            640,
            360,
        )

        with app.Image.open(animation) as image:
            self.assertEqual(image.size, (640, 360))
            self.assertGreater(image.n_frames, 1)

    def test_huggingface_creates_six_animated_scenes(self):
        transcript = " ".join(f"story-word-{index}" for index in range(120))
        work_dir = Path(self.temp_dir.name) / "huggingface-scenes"
        work_dir.mkdir()
        generated_scenes = []

        def fake_generate(_scene, _transcript, index, _count, destination, *_args):
            generated_scenes.append(index)
            destination.write_bytes(b"image")

        def fake_animate(_image, destination, *_args):
            destination.touch()

        with (
            patch.object(app, "generate_huggingface_scene_image", side_effect=fake_generate),
            patch.object(app, "create_animated_scene_gif", side_effect=fake_animate),
        ):
            image_count = len(app.generate_huggingface_animations(
                transcript,
                work_dir,
                1280,
                720,
                "hf-test-token",
                lambda _message, _progress: None,
            ))

        self.assertEqual(image_count, 6)
        self.assertEqual(len(generated_scenes), 6)

    def test_audio_image_slideshow_cycles_and_matches_audio_duration(self):
        work_dir = Path(self.temp_dir.name) / "audio-images"
        work_dir.mkdir()
        image_paths = []
        for index in range(2):
            image_path = work_dir / f"source-{index}.jpg"
            app.Image.new("RGB", (320, 180), color="blue").save(image_path)
            image_paths.append(image_path)
        audio_path = work_dir / "audio.mp3"
        audio_path.write_bytes(b"audio")
        commands = []

        with (
            patch.object(app, "run", side_effect=lambda command: commands.append(command)),
            patch.object(app, "ffprobe_duration", return_value=20),
        ):
            app.create_assets(
                "",
                "audio_images",
                "youtube",
                "en_female",
                "natural",
                "fit",
                image_paths,
                None,
                audio_path,
                work_dir,
                lambda _message, _progress: None,
            )

        video_command = commands[-2]
        self.assertIn("concat=n=4", video_command[video_command.index("-filter_complex") + 1])
        self.assertIn("zoompan=", video_command[video_command.index("-filter_complex") + 1])
        self.assertEqual(video_command.count("-i"), 5)
        output_duration_index = len(video_command) - 1 - video_command[::-1].index("-t")
        self.assertEqual(video_command[output_duration_index + 1], "20.000000")

    def test_video_upload_is_chunked_and_consumed_by_generation_job(self):
        original_chunk_size = app.VIDEO_CHUNK_BYTES
        app.VIDEO_CHUNK_BYTES = 5
        start = self.client.post(
            "/uploads",
            json={"filename": "large-video.mp4", "size": 11},
        )
        self.assertEqual(start.status_code, 201)
        upload_id = start.json["upload_id"]
        try:
            for index, chunk_body in enumerate((b"video", b"-byte", b"s")):
                chunk = self.client.post(
                    f"/uploads/{upload_id}/chunks/{index}",
                    data=chunk_body,
                    content_type="application/octet-stream",
                )
                self.assertEqual(chunk.status_code, 200)
            self.assertTrue(chunk.json["complete"])

            with patch.object(app, "Thread") as thread_class:
                response = self.client.post(
                    "/generate",
                    data={
                        "transcript": "A short test transcript.",
                        "mode": "video",
                        "platform": "youtube",
                        "video_upload_id": upload_id,
                    },
                    headers={"Accept": "application/json"},
                )

            self.assertEqual(response.status_code, 202)
            uploaded_video = thread_class.call_args.kwargs["args"][8]
            self.assertEqual(uploaded_video.read_bytes(), b"video-bytes")
            self.assertNotIn(upload_id, app.upload_sessions)
        finally:
            app.VIDEO_CHUNK_BYTES = original_chunk_size

    def test_video_upload_rejects_out_of_order_chunks_and_oversized_file(self):
        original_max = app.MAX_VIDEO_UPLOAD_BYTES
        app.MAX_VIDEO_UPLOAD_BYTES = 20
        try:
            too_large = self.client.post(
                "/uploads",
                json={"filename": "large-video.mp4", "size": 21},
            )
            self.assertEqual(too_large.status_code, 413)

            start = self.client.post(
                "/uploads",
                json={"filename": "video.mp4", "size": 12},
            )
            self.assertEqual(start.status_code, 201)
            upload_id = start.json["upload_id"]
            out_of_order = self.client.post(
                f"/uploads/{upload_id}/chunks/1",
                data=b"video-bytes",
                content_type="application/octet-stream",
            )
            self.assertEqual(out_of_order.status_code, 409)
        finally:
            app.MAX_VIDEO_UPLOAD_BYTES = original_max

    def test_unknown_job_returns_not_found(self):
        response = self.client.get("/jobs/not-a-job")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Generation job not found", response.json["error"])

    def test_completed_outputs_are_recovered_after_server_restart(self):
        job_id = uuid4().hex
        job_dir = app.OUTPUT_DIR / job_id
        job_dir.mkdir()
        for filename in ("video.mp4", "slideshow.gif", "audio.wav"):
            (job_dir / filename).write_bytes(b"test")
        (job_dir / "job.json").write_text(json.dumps({
            "status": "running",
            "progress": 88,
            "message": "Creating looping GIF.",
            "logs": [],
            "platform": "YouTube 16:9 (HD)",
            "voice": "English (US), female - LJSpeech High",
        }), encoding="utf-8")

        with app.jobs_lock:
            app.jobs.clear()

        status = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json["status"], "complete")
        result = self.client.get(f"/results/{job_id}")
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"Download MP4", result.data)


if __name__ == "__main__":
    unittest.main()
