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
        self.assertIn(b"stick-figure explainer scenes", response.data)
        self.assertIn(b'value="transcript_only"', response.data)
        self.assertIn(b'value="images"', response.data)
        self.assertIn(b'value="video"', response.data)
        self.assertIn(b"Instagram Feed portrait", response.data)
        self.assertIn(b'value="hi">Hindi', response.data)
        self.assertIn(b'value="fill">Fill', response.data)
        self.assertIn(b"id=\"generation-log\"", response.data)
        self.assertNotIn(b'href="/docs"', response.data)

    def test_processing_guide_is_available_from_the_app(self):
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Video Generator: flow, models, and libraries", response.data)
        self.assertIn(b"Animated poses, props, effects", response.data)
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

    def test_video_framing_preserves_image_by_default_and_supports_fill(self):
        fit_filter = app.frame_filter(720, 1280, "fit")
        fill_filter = app.frame_filter(720, 1280, "fill")

        self.assertIn("force_original_aspect_ratio=decrease", fit_filter)
        self.assertIn("pad=720:1280", fit_filter)
        self.assertNotIn("crop=", fit_filter)
        self.assertIn("force_original_aspect_ratio=increase", fill_filter)
        self.assertIn("crop=720:1280", fill_filter)

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

        self.assertEqual(len(animations), 2)
        with app.Image.open(animations[0]) as animation:
            self.assertEqual(animation.size, (640, 360))
            self.assertGreater(animation.n_frames, 1)
        self.assertIn("idea, wave, success motions", messages[1])

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
