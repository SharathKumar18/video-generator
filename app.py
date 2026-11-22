import json
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Callable
from urllib.request import urlopen

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
VOICE_DIR = BASE_DIR / "voices"
VOICE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_MODELS = {
    "en_female": {
        "label": "English (US), female - LJSpeech High",
        "filename": "en_US-ljspeech-high.onnx",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
        "license": "Public domain dataset",
    },
    "en_male": {
        "label": "English (US), male - Norman Medium",
        "filename": "en_US-norman-medium.onnx",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_US/norman/medium/en_US-norman-medium.onnx",
        "license": "Public domain dataset",
    },
    "hi_female": {
        "label": "Hindi (India), female - Priyamvada Medium",
        "filename": "hi_IN-priyamvada-medium.onnx",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx",
        "license": "CC BY-NC-SA 4.0; non-commercial use only",
    },
    "hi_male": {
        "label": "Hindi (India), male - Pratham Medium",
        "filename": "hi_IN-pratham-medium.onnx",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
        "license": "CC BY-NC-SA 4.0; non-commercial use only",
    },
}
DEFAULT_LANGUAGE = "en"
DEFAULT_GENDER = "female"
TONES = {
    "natural": {"label": "Natural", "length_scale": 1.0},
    "calm": {"label": "Calm / measured", "length_scale": 1.12},
    "upbeat": {"label": "Upbeat / brisk", "length_scale": 0.92},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "250")) * 1024 * 1024

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
FORMATS = {
    "youtube": {"label": "YouTube 16:9 (HD)", "width": 1280, "height": 720},
    "vertical": {"label": "Shorts / Reels 9:16", "width": 720, "height": 1280},
    "instagram": {"label": "Instagram Feed 4:5", "width": 864, "height": 1080},
    "square": {"label": "Square 1:1", "width": 1080, "height": 1080},
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}
jobs_lock = Lock()
generation_lock = Lock()

def run(command: list[str]) -> None:
    if Path(command[0]).name == "ffmpeg":
        command = [command[0], "-hide_banner", "-loglevel", "error", *command[1:]]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"FFmpeg failed: {detail[-1500:]}") from exc


def ensure_voice(voice_id: str, log: Callable[[str, int], None]) -> Path:
    voice = VOICE_MODELS[voice_id]
    model_path = VOICE_DIR / voice["filename"]
    config_path = Path(f"{model_path}.json")
    if model_path.exists() and config_path.exists():
        return model_path

    log(f"Downloading Piper voice: {voice['label']} (first use only).", 8)
    for destination, url in (
        (model_path, f"{voice['url']}?download=true"),
        (config_path, f"{voice['url']}.json?download=true"),
    ):
        if destination.exists():
            continue
        temporary_path = destination.with_suffix(f"{destination.suffix}.part")
        try:
            with urlopen(url, timeout=60) as response:
                with temporary_path.open("wb") as model_file:
                    shutil.copyfileobj(response, model_file)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return model_path


def create_speech(
    transcript: str,
    audio_path: Path,
    voice_id: str,
    tone: str,
    log: Callable[[str, int], None],
) -> None:
    piper = shutil.which("piper")
    if piper is None:
        raise FileNotFoundError("Piper is missing. Install the project requirements and restart.")
    model_path = ensure_voice(voice_id, log)
    log(
        f"Generating speech with Piper: {VOICE_MODELS[voice_id]['label']}; "
        f"delivery {TONES[tone]['label'].lower()}.",
        10,
    )
    subprocess.run(
        [
            piper,
            "--model",
            str(model_path),
            "--output_file",
            str(audio_path),
            "--length-scale",
            str(TONES[tone]["length_scale"]),
        ],
        input=transcript,
        text=True,
        check=True,
    )


def log_job(
    job_id: str,
    message: str,
    progress: int | None = None,
    status: str | None = None,
) -> None:
    entry = {
        "time": datetime.now().astimezone().strftime("%H:%M:%S"),
        "message": message,
    }
    with jobs_lock:
        job = jobs[job_id]
        job["logs"].append(entry)
        job["message"] = message
        if progress is not None:
            job["progress"] = progress
        if status is not None:
            job["status"] = status
        snapshot = {**job, "logs": list(job["logs"])}
    persist_job(job_id, snapshot)
    print(f"[{job_id}] {entry['time']} {message}", flush=True)


def persist_job(job_id: str, job: dict) -> None:
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.is_dir():
        return
    record_path = job_dir / "job.json"
    temporary_path = job_dir / "job.json.part"
    temporary_path.write_text(json.dumps(job), encoding="utf-8")
    temporary_path.replace(record_path)


def existing_output_files(job_dir: Path) -> dict[str, str]:
    candidates = {
        "mp4": "video.mp4",
        "gif": "slideshow.gif",
        "audio": "audio.wav",
    }
    return {
        key: filename
        for key, filename in candidates.items()
        if (job_dir / filename).is_file()
    }


def read_job(job_id: str) -> dict | None:
    try:
        uuid.UUID(job_id)
    except ValueError:
        return None

    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            return {**job, "logs": list(job["logs"])}

    job_dir = OUTPUT_DIR / job_id
    record_path = job_dir / "job.json"
    if record_path.is_file():
        try:
            job = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "failed",
                "progress": 0,
                "message": f"Could not read the saved job log: {exc}",
                "logs": [],
                "files": existing_output_files(job_dir),
            }

        files = existing_output_files(job_dir)
        if job["status"] in {"queued", "running"}:
            complete = {"mp4", "gif", "audio"}.issubset(files)
            job["status"] = "complete" if complete else "failed"
            job["progress"] = 100 if complete else job.get("progress", 0)
            job["message"] = (
                "Recovered completed outputs after a server restart."
                if complete
                else "The server restarted before this job completed; available partial outputs are retained."
            )
            job["files"] = files
            job["logs"].append(
                {
                    "time": datetime.now().astimezone().strftime("%H:%M:%S"),
                    "message": job["message"],
                }
            )
            persist_job(job_id, job)
        return job

    files = existing_output_files(job_dir)
    if not files:
        return None
    complete = {"mp4", "gif", "audio"}.issubset(files)
    message = (
        "Recovered completed files from before persistent job logs were enabled."
        if complete
        else "The server stopped before this job completed; available partial outputs are retained."
    )
    return {
        "status": "complete" if complete else "failed",
        "progress": 100 if complete else 0,
        "message": message,
        "logs": [
            {
                "time": datetime.now().astimezone().strftime("%H:%M:%S"),
                "message": message,
            }
        ],
        "files": files,
        "platform": "Unknown (created by an earlier server version)",
        "voice": None,
    }


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def generate_stickman_images(
    transcript: str,
    work_dir: Path,
    width: int,
    height: int,
    log: Callable[[str, int], None],
) -> list[Path]:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?।])\s+", transcript)
        if part.strip()
    ] or [transcript.strip()]
    scene_count = min(8, len(sentences))
    chunk_size = (len(sentences) + scene_count - 1) // scene_count
    scenes = [
        " ".join(sentences[index:index + chunk_size])
        for index in range(0, len(sentences), chunk_size)
    ]
    log(f"Drawing {len(scenes)} stick-figure scenes from the transcript locally.", 25)

    scale = min(width, height) / 720
    stroke = max(3, round(6 * scale))
    paths: list[Path] = []
    for index, sentence in enumerate(scenes):
        image = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, int(height * 0.83), width, height), fill="#e2e8f0")
        draw.ellipse(
            (width * 0.08, height * 0.09, width * 0.19, height * 0.25),
            fill="#dbeafe",
        )

        cx = int(width * 0.34)
        baseline = int(height * 0.78)
        figure_height = int(min(height * 0.49, width * 0.62))
        head_radius = max(12, int(figure_height * 0.10))
        head_y = baseline - figure_height
        torso_top = head_y + head_radius * 2 + stroke
        torso_bottom = baseline - int(figure_height * 0.38)
        draw.ellipse(
            (
                cx - head_radius,
                head_y,
                cx + head_radius,
                head_y + head_radius * 2,
            ),
            outline="#172554",
            width=stroke,
        )
        draw.line((cx, torso_top, cx, torso_bottom), fill="#172554", width=stroke)

        lowered = sentence.casefold()
        if any(word in lowered for word in ("challenge", "problem", "difficult", "obstacle", "fail", "मुश्किल", "समस्या", "चुनौती")):
            action = "challenge"
        elif any(word in lowered for word in ("team", "together", "help", "support", "साथ", "मदद", "टीम")):
            action = "team"
        elif any(word in lowered for word in ("idea", "think", "plan", "solution", "सोच", "विचार", "योजना")):
            action = "idea"
        elif any(word in lowered for word in ("grow", "growth", "learn", "improve", "progress", "बढ़", "सीख", "विकास")):
            action = "growth"
        elif any(word in lowered for word in ("goal", "success", "achieve", "win", "सफल", "लक्ष्य", "जीत")):
            action = "success"
        elif any(word in lowered for word in ("time", "deadline", "समय", "घड़ी")):
            action = "time"
        else:
            action = ("idea", "growth", "challenge", "success")[index % 4]

        arm_y = torso_top + int(figure_height * 0.17)
        arm_reach = int(figure_height * 0.24)
        if action in {"growth", "success", "idea"}:
            arms = [
                (cx - arm_reach, arm_y - arm_reach),
                (cx + arm_reach, arm_y - arm_reach),
            ]
        elif action == "challenge":
            arms = [(cx - arm_reach, arm_y), (cx + arm_reach, arm_y - arm_reach // 3)]
        else:
            arms = [(cx - arm_reach, arm_y - arm_reach // 3), (cx + arm_reach, arm_y - arm_reach // 3)]
        draw.line((cx, arm_y, *arms[0]), fill="#172554", width=stroke)
        draw.line((cx, arm_y, *arms[1]), fill="#172554", width=stroke)
        leg_y = torso_bottom
        draw.line(
            (cx, leg_y, cx - int(figure_height * 0.16), baseline),
            fill="#172554",
            width=stroke,
        )
        draw.line(
            (cx, leg_y, cx + int(figure_height * 0.16), baseline),
            fill="#172554",
            width=stroke,
        )

        ox = int(width * 0.72)
        oy = int(height * 0.49)
        object_size = max(32, int(figure_height * 0.31))
        blue = "#2563eb"
        orange = "#f97316"
        if action == "challenge":
            draw.polygon(
                [
                    (ox - object_size, baseline),
                    (ox, baseline - object_size),
                    (ox + object_size, baseline),
                ],
                fill="#fb923c",
            )
        elif action == "team":
            other_x = int(width * 0.80)
            radius = max(10, head_radius * 2 // 3)
            draw.ellipse((other_x - radius, oy - object_size, other_x + radius, oy - object_size + radius * 2), outline=blue, width=stroke)
            draw.line((other_x, oy - object_size + radius * 2, other_x, baseline), fill=blue, width=stroke)
            draw.line((cx + arm_reach, arm_y - arm_reach // 3, other_x - radius, arm_y), fill=blue, width=stroke)
            draw.line((other_x, baseline, other_x - radius, baseline + stroke * 3), fill=blue, width=stroke)
            draw.line((other_x, baseline, other_x + radius, baseline + stroke * 3), fill=blue, width=stroke)
        elif action == "idea":
            draw.ellipse((ox - object_size // 2, oy - object_size, ox + object_size // 2, oy), outline=orange, width=stroke)
            draw.line((ox - object_size // 4, oy + stroke, ox + object_size // 4, oy + stroke), fill=orange, width=stroke)
            for ray in range(8):
                angle = ray * math.pi / 4
                x1 = ox + int(math.cos(angle) * object_size * 0.68)
                y1 = oy - object_size // 2 + int(math.sin(angle) * object_size * 0.68)
                x2 = ox + int(math.cos(angle) * object_size * 0.9)
                y2 = oy - object_size // 2 + int(math.sin(angle) * object_size * 0.9)
                draw.line((x1, y1, x2, y2), fill=orange, width=stroke)
        elif action == "growth":
            for step in range(4):
                x1 = ox - object_size + step * object_size // 2
                y1 = baseline - step * object_size // 3
                draw.rectangle((x1, y1, x1 + object_size // 2, baseline), fill="#93c5fd")
            draw.line((ox + object_size, baseline - object_size, ox + object_size, baseline - object_size * 2), fill=blue, width=stroke)
            draw.polygon(((ox + object_size, baseline - object_size * 2), (ox + object_size * 2, baseline - object_size * 2 + stroke * 3), (ox + object_size, baseline - object_size * 2 + stroke * 6)), fill=orange)
        elif action == "success":
            draw.rectangle((ox - object_size // 2, oy, ox + object_size // 2, oy + object_size // 2), fill="#fbbf24")
            draw.rectangle((ox - object_size, oy + object_size // 2, ox + object_size, oy + object_size), fill="#fbbf24")
            draw.rectangle((ox - object_size // 3, oy + object_size, ox + object_size // 3, oy + object_size + stroke * 3), fill=orange)
        elif action == "time":
            draw.ellipse((ox - object_size, oy - object_size, ox + object_size, oy + object_size), outline=blue, width=stroke)
            draw.line((ox, oy, ox, oy - object_size // 2), fill=blue, width=stroke)
            draw.line((ox, oy, ox + object_size // 2, oy + object_size // 3), fill=blue, width=stroke)

        path = work_dir / f"stickman-scene-{index:03d}.png"
        image.save(path)
        log(f"Created stick-figure scene {index + 1} of {len(scenes)}.", 26 + round(16 * (index + 1) / len(scenes)))
        paths.append(path)
    return paths


def frame_filter(width: int, height: int, layout: str) -> str:
    if layout == "fill":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white"
    )


def normalize_image(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    layout: str,
) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            frame_filter(width, height, layout),
            "-frames:v",
            "1",
            str(destination),
        ]
    )


def create_assets(
    transcript: str,
    mode: str,
    platform: str,
    voice_id: str,
    tone: str,
    layout: str,
    image_paths: list[Path],
    video_path: Path | None,
    work_dir: Path,
    log: Callable[[str, int], None],
) -> tuple[Path, Path, Path]:
    transcript_path = work_dir / "transcript.txt"
    audio_path = work_dir / "audio.wav"
    mp4_path = work_dir / "video.mp4"
    gif_path = work_dir / "slideshow.gif"
    transcript_path.write_text(transcript, encoding="utf-8")

    create_speech(transcript, audio_path, voice_id, tone, log)
    audio_duration = ffprobe_duration(audio_path)
    log(f"Speech audio created; duration is {audio_duration:.1f} seconds.", 22)

    output_format = FORMATS[platform]
    width = output_format["width"]
    height = output_format["height"]
    log(f"Output profile: {output_format['label']} ({width}x{height}).", 23)

    if mode == "transcript_only":
        image_paths = generate_stickman_images(transcript, work_dir, width, height, log)

    if mode == "video" and video_path is not None:
        log("Using the uploaded video as the visual track and replacing its original audio.", 45)
        log("Encoding the video to the selected profile and narration duration.", 60)
        run(
            [
                "ffmpeg",
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-vf",
                f"{frame_filter(width, height, layout)},"
                "setsar=1,fps=30,format=yuv420p",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{audio_duration:.6f}",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
        )
    else:
        log(f"Preparing {len(image_paths)} slideshow images for {output_format['label']}.", 45)
        normalized_images = []
        for index, image_path in enumerate(image_paths):
            progress = 46 + round(8 * index / len(image_paths))
            log(f"Resizing image {index + 1} of {len(image_paths)} to {width}x{height}.", progress)
            normalized_path = work_dir / f"image-{index:03d}.jpg"
            normalize_image(image_path, normalized_path, width, height, layout)
            normalized_images.append(normalized_path)

        segment_duration = audio_duration / len(normalized_images)
        log(
            f"Timing {len(normalized_images)} images for about {segment_duration:.1f} seconds each.",
            55,
        )
        command = ["ffmpeg", "-y"]
        for image_path in normalized_images:
            command.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-t",
                    f"{segment_duration:.6f}",
                    "-i",
                    str(image_path),
                ]
            )

        filters = []
        for index in range(len(normalized_images)):
            filters.append(
                f"[{index}:v]scale={width}:{height},setsar=1,"
                f"fps=30,format=yuv420p[v{index}]"
            )
        concat_inputs = "".join(f"[v{index}]" for index in range(len(normalized_images)))
        filters.append(
            f"{concat_inputs}concat=n={len(normalized_images)}:v=1:a=0,"
            "format=yuv420p[outv]"
        )

        command.extend(
            [
                "-i",
                str(audio_path),
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[outv]",
                "-map",
                f"{len(normalized_images)}:a",
                "-c:v",
                "libx264",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{audio_duration:.6f}",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
        )
        log("Encoding MP4 video (H.264) with AAC narration using FFmpeg.", 60)
        run(command)

    log("Creating looping GIF from the generated MP4.", 88)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            "fps=6,scale=480:-1:flags=lanczos,"
            "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(gif_path),
        ]
    )
    log("MP4, GIF, and WAV files are ready.", 98)
    return mp4_path, gif_path, audio_path


def generate_job(
    job_id: str,
    transcript: str,
    mode: str,
    platform: str,
    voice_id: str,
    tone: str,
    layout: str,
    image_paths: list[Path],
    video_path: Path | None,
    work_dir: Path,
) -> None:
    log_job(job_id, "Job is queued; waiting for the generation slot.", status="queued")
    with generation_lock:
        log_job(
            job_id,
            f"Generation started with Piper voice: {VOICE_MODELS[voice_id]['label']}.",
            status="running",
        )
        try:
            mp4_path, gif_path, audio_path = create_assets(
                transcript,
                mode,
                platform,
                voice_id,
                tone,
                layout,
                image_paths,
                video_path,
                work_dir,
                lambda message, progress: log_job(job_id, message, progress),
            )
        except (subprocess.CalledProcessError, ImportError, OSError, RuntimeError, ValueError) as exc:
            files = existing_output_files(work_dir)
            with jobs_lock:
                jobs[job_id]["files"] = files
            log_job(job_id, f"Generation failed: {exc}", status="failed")
            return

        with jobs_lock:
            jobs[job_id]["files"] = {
                "mp4": mp4_path.name,
                "gif": gif_path.name,
                "audio": audio_path.name,
            }
        log_job(job_id, "Generation complete.", progress=100, status="complete")


@app.get("/")
def index():
    return render_template("index.html")


@app.errorhandler(413)
def request_entity_too_large(_error):
    max_upload_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    if request.accept_mimetypes.best == "application/json":
        return jsonify(error=f"Combined upload exceeds the {max_upload_mb} MB server limit."), 413
    return render_template(
        "upload_too_large.html",
        max_upload_mb=max_upload_mb,
    ), 413


@app.post("/generate")
def generate():
    transcript = request.form.get("transcript", "").strip()
    mode = request.form.get("mode", "images")
    platform = request.form.get("platform", "youtube")
    language = request.form.get("language", DEFAULT_LANGUAGE)
    gender = request.form.get("gender", DEFAULT_GENDER)
    voice_id = f"{language}_{gender}"
    tone = request.form.get("tone", "natural")
    layout = request.form.get("layout", "fit")
    uploads = request.files.getlist("images")

    if not transcript:
        return jsonify(error="Enter a transcript."), 400
    if mode not in {"transcript_only", "images", "video"}:
        return jsonify(error="Choose a valid generation mode."), 400
    if platform not in FORMATS:
        return jsonify(error="Choose a valid output format."), 400
    if language not in {"en", "hi"} or gender not in {"female", "male"}:
        return jsonify(error="Choose a valid narration language and voice."), 400
    if voice_id not in VOICE_MODELS:
        return jsonify(error="Choose a valid speech voice."), 400
    if tone not in TONES:
        return jsonify(error="Choose a valid delivery pace."), 400
    if layout not in {"fit", "fill"}:
        return jsonify(error="Choose whether to fit or crop the visual."), 400

    valid_uploads = []
    image_paths = []
    video_path = None

    if mode == "images":
        if not uploads or all(not upload.filename for upload in uploads):
            return jsonify(error="Upload at least one image."), 400
        for upload in uploads:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                return jsonify(error="Only JPG, PNG, and WebP images are supported."), 400
            valid_uploads.append((upload, suffix))
        for upload, suffix in valid_uploads:
            image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
            upload.save(image_path)
            image_paths.append(image_path)
    elif mode == "video":
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return jsonify(error="Upload a video file."), 400
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify(error="Use an MP4, MOV, M4V, or WebM video."), 400
        video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        upload.save(video_path)

    job_id = uuid.uuid4().hex
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Generation queued.",
            "logs": [],
            "platform": FORMATS[platform]["label"],
            "voice": VOICE_MODELS[voice_id]["label"],
            "language": language,
            "tone": TONES[tone]["label"],
            "mode": mode,
        }
    uploaded_paths = image_paths + ([video_path] if video_path else [])
    total_size = sum(path.stat().st_size for path in uploaded_paths)
    media_description = (
        f"{len(image_paths)} images"
        if mode == "images"
        else "one video" if mode == "video" else "transcript-only stick-figure visuals"
    )
    log_job(
        job_id,
        f"Received {media_description} ({total_size / (1024 * 1024):.1f} MB); "
        f"output profile: {FORMATS[platform]['label']}.",
        status="queued",
    )
    Thread(
        target=generate_job,
        args=(
            job_id,
            transcript,
            mode,
            platform,
            voice_id,
            tone,
            layout,
            image_paths,
            video_path,
            work_dir,
        ),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id), 202


@app.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = read_job(job_id)
    if job is None:
        return jsonify(error="Generation job not found."), 404
    return jsonify(job)


@app.get("/results/<job_id>")
def job_results(job_id: str):
    job = read_job(job_id)
    if job is None or job["status"] not in {"complete", "failed"}:
        return "Generation results are not ready.", 404
    files = job.get("files", {})
    return render_template(
        "index.html",
        job_id=job_id,
        mp4_name=files.get("mp4"),
        gif_name=files.get("gif"),
        audio_name=files.get("audio"),
        logs=job.get("logs", []),
        platform=job.get("platform"),
        voice=job.get("voice"),
        tone=job.get("tone"),
        error=job["message"] if job["status"] == "failed" else None,
    )


@app.get("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    return send_from_directory(OUTPUT_DIR / job_id, filename, as_attachment=True)


@app.get("/preview/<job_id>/<filename>")
def preview(job_id: str, filename: str):
    return send_from_directory(OUTPUT_DIR / job_id, filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
