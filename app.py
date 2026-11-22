import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageOps


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DATA_DIR / "output"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", DATA_DIR / "uploads"))
VOICE_DIR = Path(os.environ.get("VOICE_DIR", DATA_DIR / "voices"))
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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
MAX_VIDEO_UPLOAD_MB = int(
    os.environ.get("MAX_VIDEO_UPLOAD_MB", os.environ.get("MAX_UPLOAD_MB", "2048"))
)
MAX_VIDEO_UPLOAD_BYTES = MAX_VIDEO_UPLOAD_MB * 1024 * 1024
VIDEO_CHUNK_BYTES = 8 * 1024 * 1024

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
FORMATS = {
    "youtube": {"label": "YouTube 16:9 (HD)", "width": 1280, "height": 720},
    "vertical": {"label": "Shorts / Reels 9:16", "width": 720, "height": 1280},
    "instagram": {"label": "Instagram Feed 4:5", "width": 864, "height": 1080},
    "square": {"label": "Square 1:1", "width": 1080, "height": 1080},
}
GENERATION_MODES = {
    "transcript_only": {
        "title": "Transcript only",
        "description": "Create a narrated video with illustrated scenes. Add a Gemini key for AI-generated artwork.",
        "icon": "✦",
    },
    "images": {
        "title": "Transcript + images",
        "description": "Narrate your transcript over an ordered sequence of images.",
        "icon": "▧",
    },
    "audio_images": {
        "title": "Existing audio + images",
        "description": "Use your audio track and cycle through images with gentle motion.",
        "icon": "♫",
    },
    "video": {
        "title": "Transcript + video",
        "description": "Replace the source video's audio with narration from your transcript.",
        "icon": "▶",
    },
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}
jobs_lock = Lock()
generation_lock = Lock()
upload_sessions: dict[str, dict] = {}
upload_sessions_lock = Lock()

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
    }
    files = {
        key: filename
        for key, filename in candidates.items()
        if (job_dir / filename).is_file()
    }
    audio_path = next(
        (path for path in job_dir.glob("audio.*") if path.is_file()),
        None,
    )
    if audio_path is not None:
        files["audio"] = audio_path.name
    return files


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
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        result = subprocess.run(
            [
                ffprobe,
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

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is missing. Install FFmpeg and restart the app.")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr,
    )
    if match is None:
        raise RuntimeError(f"Could not determine media duration: {result.stderr.strip()[-1500:]}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


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


def transcript_scenes(transcript: str, scene_count: int) -> list[str]:
    words = transcript.split()
    scene_count = max(1, scene_count)
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?।])\s+", transcript)
        if part.strip()
    ] or [transcript.strip()]
    if len(sentences) >= scene_count:
        return [
            " ".join(
                sentences[index * len(sentences) // scene_count:
                          (index + 1) * len(sentences) // scene_count]
            )
            for index in range(scene_count)
        ]

    scenes = [
        " ".join(
            words[index * len(words) // scene_count:
                  (index + 1) * len(words) // scene_count]
        )
        for index in range(scene_count)
    ]
    return [scene or transcript for scene in scenes]


def generate_gemini_scene_image(
    scene: str,
    transcript: str,
    scene_index: int,
    scene_count: int,
    destination: Path,
    api_key: str,
    character_reference: bytes | None = None,
) -> None:
    prompt = (
        "Create one polished, cinematic 16:9 storybook illustration for a narrated video. "
        "Use expressive characters, rich color, clear composition, and a consistent "
        "hand-painted editorial animation style. Do not include words, lettering, "
        "captions, logos, borders, or watermarks. "
        f"Create scene {scene_index + 1} of {scene_count} in a distinct, consistent "
        "storyboard sequence. Illustrate this narration "
        f"moment: {scene}. Story context: {transcript}"
    )
    if character_reference:
        prompt += (
            ". Use the person or character in the supplied reference image as the "
            "recurring main character. Preserve their recognizable appearance."
        )
    content: list[dict] = [{"type": "text", "text": prompt}]
    if character_reference:
        content.append({
            "type": "image",
            "mime_type": "image/jpeg",
            "data": base64.b64encode(character_reference).decode("ascii"),
        })

    body = json.dumps({
        "model": "gemini-3.1-flash-lite-image",
        "input": content,
        "response_format": {
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": "16:9",
            "image_size": "1024px",
        },
    }).encode("utf-8")
    api_request = Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Api-Revision": "2026-05-20",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(api_request, timeout=120) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(
            f"Gemini image generation failed (HTTP {exc.code}). "
            "Check the API key, model access, and quota."
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Gemini: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned an invalid image response.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an invalid image response.")
    image_data = None
    output_image = result.get("output_image")
    if isinstance(output_image, dict):
        image_data = output_image.get("data")
    if not image_data:
        outputs = result.get("output", [])
        if not isinstance(outputs, list):
            raise RuntimeError("Gemini returned an invalid image response.")
        for output in outputs:
            if isinstance(output, dict) and output.get("type") == "image":
                image_data = output.get("data")
                if image_data:
                    break
    if not isinstance(image_data, str):
        raise RuntimeError("Gemini did not return an image for a transcript scene.")
    try:
        if image_data.startswith("data:") and "," in image_data:
            image_data = image_data.split(",", 1)[1]
        destination.write_bytes(base64.b64decode(image_data, validate=True))
    except (ValueError, OSError) as exc:
        raise RuntimeError("Gemini returned invalid image data.") from exc


def create_animated_scene_gif(
    image_path: Path,
    destination: Path,
    scene_text: str,
    width: int,
    height: int,
) -> None:
    with Image.open(image_path) as source:
        scene_image = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )

    frames = []
    for frame_index in range(24):
        phase = frame_index * 2 * math.pi / 24
        zoom = 1.0 + 0.045 * (0.5 - 0.5 * math.cos(phase))
        scaled = scene_image.resize(
            (round(width * zoom), round(height * zoom)),
            Image.Resampling.LANCZOS,
        )
        pan = math.sin(phase) * width * 0.012
        left = round((scaled.width - width) / 2 + pan)
        top = (scaled.height - height) // 2
        frame = scaled.crop((left, top, left + width, top + height))
        draw = ImageDraw.Draw(frame, "RGBA")
        panel_height = max(52, round(height * 0.20))
        draw.rounded_rectangle(
            (
                round(width * 0.035),
                round(height * 0.035),
                round(width * 0.78),
                panel_height,
            ),
            radius=max(8, round(height * 0.025)),
            fill=(15, 23, 42, 205),
        )
        caption_font = ImageFont.load_default(size=max(14, round(height * 0.045)))
        caption = "\n".join(textwrap.wrap(scene_text, width=max(24, width // 18))[:2])
        draw.multiline_text(
            (round(width * 0.06), round(height * 0.055)),
            caption,
            fill="#ffffff",
            font=caption_font,
            spacing=3,
        )
        frames.append(frame)

    palette = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    indexed_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    indexed_frames[0].save(
        destination,
        save_all=True,
        append_images=indexed_frames[1:],
        duration=1000 // 12,
        loop=0,
        optimize=False,
    )


def generate_gemini_animations(
    transcript: str,
    work_dir: Path,
    width: int,
    height: int,
    has_character: bool,
    api_key: str,
    character_reference: bytes | None,
    log: Callable[[str, int], None],
) -> list[Path]:
    scene_count = 8 if has_character else 6
    scenes = transcript_scenes(transcript, scene_count)
    render_scale = min(1.0, 640 / max(width, height))
    width = max(1, round(width * render_scale))
    height = max(1, round(height * render_scale))
    def render_scene(index: int, scene: str) -> Path:
        image_path = work_dir / f"gemini-scene-{index:02d}.png"
        animation_path = work_dir / f"gemini-animation-{index:02d}.gif"
        generate_gemini_scene_image(
            scene,
            transcript,
            index,
            len(scenes),
            image_path,
            api_key,
            character_reference,
        )
        create_animated_scene_gif(
            image_path,
            animation_path,
            scene,
            width,
            height,
        )
        image_path.unlink()
        return animation_path

    log(
        f"Generating {len(scenes)} Gemini illustrations and animated loops.",
        26,
    )
    with ThreadPoolExecutor(max_workers=min(3, len(scenes))) as executor:
        paths = list(executor.map(render_scene, range(len(scenes)), scenes))
    log(f"Created {len(paths)} animated Gemini scenes.", 42)
    return paths


def generate_stickman_animations(
    transcript: str,
    work_dir: Path,
    width: int,
    height: int,
    duration: float,
    log: Callable[[str, int], None],
) -> list[Path]:
    render_scale = min(1.0, 640 / max(width, height))
    width = max(1, round(width * render_scale))
    height = max(1, round(height * render_scale))
    scenes = transcript_scenes(
        transcript,
        min(6, max(1, len(transcript.split()))),
    )
    log(f"Animating {len(scenes)} illustrated stick-figure scenes.", 25)

    frame_count = 24
    paths = []
    scale = min(width, height)
    stroke = max(7, round(scale * 0.018))
    head_radius = max(14, round(scale * 0.045))
    figure_height = round(min(height * 0.40, width * 0.30))
    captions_font = ImageFont.load_default(size=max(18, round(scale * 0.035)))
    small_font = ImageFont.load_default(size=max(14, round(scale * 0.022)))

    for scene_index, sentence in enumerate(scenes):
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
        elif any(word in lowered for word in ("hello", "welcome", "wave", "namaste", "नमस्ते")):
            action = "wave"
        else:
            action = ("wave", "idea", "growth", "challenge", "success")[scene_index % 5]

        scene_action = action
        animation_sequences = {
            "challenge": ("challenge", "growth", "success"),
            "growth": ("growth", "success", "wave"),
            "idea": ("idea", "wave", "success"),
            "success": ("wave", "success", "wave"),
            "team": ("team", "wave", "success"),
            "time": ("time", "challenge", "success"),
            "wave": ("wave", "idea", "success"),
        }
        action_sequence = animation_sequences[scene_action]
        frames = []
        for frame_index in range(frame_count):
            action = action_sequence[min(2, frame_index * 3 // frame_count)]
            phase = 2 * math.pi * frame_index / frame_count
            image = Image.new("RGB", (width, height), "#eaf4ff")
            draw = ImageDraw.Draw(image)
            for y in range(0, height, max(1, height // 40)):
                blend = y / max(1, height)
                color = (
                    round(219 - 15 * blend),
                    round(238 - 5 * blend),
                    round(255 - 1 * blend),
                )
                draw.rectangle((0, y, width, min(height, y + max(1, height // 40))), fill=color)

            draw.ellipse(
                (width * 0.77, height * 0.10, width * 0.88, height * 0.27),
                fill="#fde68a",
            )
            cloud_x = width * (0.50 + 0.025 * math.sin(phase))
            draw.ellipse((cloud_x, height * 0.20, cloud_x + width * 0.09, height * 0.27), fill="#ffffff")
            draw.ellipse((cloud_x + width * 0.035, height * 0.16, cloud_x + width * 0.10, height * 0.27), fill="#ffffff")
            draw.ellipse((width * 0.02, height * 0.66, width * 0.55, height * 0.91), fill="#bfdbfe")
            draw.ellipse((width * 0.43, height * 0.64, width * 1.02, height * 0.92), fill="#a7f3d0")
            baseline = round(height * 0.79)
            draw.rectangle((0, baseline, width, height), fill="#d1fae5")
            draw.line((0, baseline, width, baseline), fill="#86b9a2", width=max(2, stroke // 2))

            panel = (width * 0.055, height * 0.045, width * 0.66, height * 0.23)
            draw.rounded_rectangle(panel, radius=max(12, stroke * 3), fill="#ffffff", outline="#cbd5e1", width=2)
            draw.text((panel[0] + width * 0.025, panel[1] + height * 0.025), f"SCENE {scene_index + 1:02d}", font=small_font, fill="#2563eb")
            caption_lines = textwrap.wrap(sentence, width=max(20, int(width / 22)))[:2]
            draw.multiline_text(
                (panel[0] + width * 0.025, panel[1] + height * 0.075),
                "\n".join(caption_lines),
                font=captions_font,
                fill="#172554",
                spacing=max(3, stroke),
            )

            cx = round(width * (0.35 + (0.012 * math.sin(phase) if action == "challenge" else 0)))
            bounce = abs(math.sin(phase)) * height * 0.025 if action in {"success", "growth"} else abs(math.sin(phase)) * height * 0.012
            foot_y = round(baseline - bounce)
            head_y = foot_y - figure_height
            head_cx = cx
            head_cy = head_y + head_radius
            torso_top = head_y + head_radius * 2 + stroke
            torso_bottom = foot_y - round(figure_height * 0.35)
            arm_y = torso_top + round(figure_height * 0.17)
            arm_reach = round(figure_height * 0.24)
            torso = "#172554"

            draw.ellipse(
                (head_cx - head_radius, head_y, head_cx + head_radius, head_y + 2 * head_radius),
                fill="#fff7ed",
                outline=torso,
                width=stroke,
            )
            draw.ellipse((head_cx - head_radius * 0.48, head_cy - head_radius * 0.12, head_cx - head_radius * 0.30, head_cy + head_radius * 0.06), fill=torso)
            draw.ellipse((head_cx + head_radius * 0.30, head_cy - head_radius * 0.12, head_cx + head_radius * 0.48, head_cy + head_radius * 0.06), fill=torso)
            if action == "success":
                draw.arc((head_cx - head_radius * 0.35, head_cy, head_cx + head_radius * 0.35, head_cy + head_radius * 0.38), 0, 180, fill="#f97316", width=max(2, stroke // 2))
            else:
                draw.arc((head_cx - head_radius * 0.28, head_cy + head_radius * 0.05, head_cx + head_radius * 0.28, head_cy + head_radius * 0.35), 0, 180, fill="#f97316", width=max(2, stroke // 2))
            draw.line((cx, torso_top, cx, torso_bottom), fill=torso, width=stroke)
            joint_radius = stroke // 2
            draw.ellipse(
                (cx - joint_radius, torso_bottom - joint_radius, cx + joint_radius, torso_bottom + joint_radius),
                fill=torso,
            )

            if action == "success":
                arms = [(cx - arm_reach, arm_y - arm_reach), (cx + arm_reach, arm_y - arm_reach)]
            elif action == "challenge":
                arms = [(cx + arm_reach, arm_y), (cx + arm_reach * 1.1, arm_y + stroke)]
            elif action == "idea":
                arms = [(cx - arm_reach, arm_y - arm_reach * 0.45), (cx + head_radius, head_cy + head_radius * 0.35)]
            elif action == "team":
                arms = [(cx - arm_reach, arm_y - arm_reach * 0.25), (cx + arm_reach, arm_y - arm_reach * 0.25)]
            elif action == "time":
                arms = [(cx - arm_reach, arm_y - arm_reach * 0.25), (cx + arm_reach, arm_y - arm_reach * 0.65)]
            elif action == "wave":
                wave_height = 0.35 + 0.6 * (0.5 + 0.5 * math.sin(phase * 2))
                arms = [(cx - arm_reach, arm_y - round(arm_reach * wave_height)), (cx + arm_reach, arm_y - arm_reach * 0.25)]
            else:
                swing = round(math.sin(phase) * arm_reach * 0.65)
                arms = [(cx - arm_reach, arm_y + swing), (cx + arm_reach, arm_y - swing)]
            for hand_x, hand_y in arms:
                elbow_x = round((cx + hand_x) / 2)
                elbow_y = round((arm_y + hand_y) / 2 + (stroke * 2 if hand_x < cx else 0))
                draw.line((cx, arm_y, elbow_x, elbow_y, hand_x, hand_y), fill=torso, width=stroke, joint="curve")
                for joint_x, joint_y in ((cx, arm_y), (elbow_x, elbow_y)):
                    draw.ellipse(
                        (joint_x - joint_radius, joint_y - joint_radius, joint_x + joint_radius, joint_y + joint_radius),
                        fill=torso,
                    )
                draw.ellipse((hand_x - stroke, hand_y - stroke, hand_x + stroke, hand_y + stroke), fill="#f97316")

            leg_swing = math.sin(phase) * figure_height * 0.11 if action in {"growth", "challenge", "wave"} else math.sin(phase * 2) * figure_height * 0.035
            for direction in (-1, 1):
                knee_x = cx + round(direction * figure_height * 0.09 + leg_swing * direction)
                knee_y = torso_bottom + round(figure_height * 0.17)
                foot_x = cx + round(direction * figure_height * 0.18 + leg_swing * direction)
                foot_y_leg = foot_y
                draw.line((cx, torso_bottom, knee_x, knee_y, foot_x, foot_y_leg), fill=torso, width=stroke, joint="curve")
                draw.ellipse(
                    (knee_x - joint_radius, knee_y - joint_radius, knee_x + joint_radius, knee_y + joint_radius),
                    fill=torso,
                )
                draw.line((foot_x, foot_y_leg, foot_x + round(figure_height * 0.07), foot_y_leg), fill=torso, width=stroke)

            prop_x = round(width * 0.76)
            prop_y = round(height * 0.50)
            prop_size = round(min(width, height) * 0.10)
            accent = "#f97316"
            if action == "idea":
                bob = round(math.sin(phase) * prop_size * 0.12)
                draw.ellipse((prop_x - prop_size // 2, prop_y - prop_size + bob, prop_x + prop_size // 2, prop_y + bob), fill="#fef3c7", outline=accent, width=stroke)
                draw.line((prop_x - prop_size // 4, prop_y + bob + stroke, prop_x + prop_size // 4, prop_y + bob + stroke), fill=accent, width=stroke)
                for ray in range(8):
                    angle = ray * math.pi / 4 + phase * 0.08
                    x1 = prop_x + int(math.cos(angle) * prop_size * 0.65)
                    y1 = prop_y - prop_size // 2 + bob + int(math.sin(angle) * prop_size * 0.65)
                    x2 = prop_x + int(math.cos(angle) * prop_size * 0.90)
                    y2 = prop_y - prop_size // 2 + bob + int(math.sin(angle) * prop_size * 0.90)
                    draw.line((x1, y1, x2, y2), fill=accent, width=max(2, stroke // 2))
            elif action == "challenge":
                block_x = prop_x - round(math.sin(phase) * prop_size * 0.12)
                draw.rounded_rectangle((block_x - prop_size, baseline - prop_size, block_x + prop_size, baseline), radius=stroke, fill="#fb923c", outline="#c2410c", width=stroke)
                draw.line((block_x, baseline - prop_size * 0.75, block_x - prop_size * 0.16, baseline - prop_size * 0.35, block_x + prop_size * 0.15, baseline - prop_size * 0.12), fill="#7c2d12", width=max(2, stroke // 2))
            elif action == "growth":
                for step in range(4):
                    x1 = prop_x - prop_size + step * prop_size // 2
                    y1 = baseline - step * prop_size // 3
                    draw.rectangle((x1, y1, x1 + prop_size // 2, baseline), fill="#60a5fa")
                draw.line((prop_x + prop_size, baseline - prop_size, prop_x + prop_size, baseline - prop_size * 2), fill="#2563eb", width=stroke)
                draw.polygon(((prop_x + prop_size, baseline - prop_size * 2), (prop_x + prop_size * 2, baseline - prop_size * 2 + stroke * 2), (prop_x + prop_size, baseline - prop_size * 2 + stroke * 4)), fill=accent)
            elif action == "success":
                for star in range(7):
                    sx = round(width * (0.58 + ((star * 37 + frame_index * 11) % 35) / 100))
                    sy = round(height * (0.30 + ((star * 23 + frame_index * 7) % 35) / 100))
                    radius = max(3, stroke * (1 if star % 2 else 2))
                    draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=("#fbbf24", "#fb7185", "#38bdf8")[star % 3])
                draw.rounded_rectangle((prop_x - prop_size, prop_y, prop_x + prop_size, prop_y + prop_size), radius=stroke, fill="#fbbf24", outline="#d97706", width=stroke)
            elif action == "team":
                other_x = round(width * 0.52)
                other_head = head_radius * 2 // 3
                other_top = baseline - figure_height * 0.76
                draw.ellipse((other_x - other_head, other_top, other_x + other_head, other_top + other_head * 2), fill="#fef3c7", outline="#2563eb", width=stroke)
                draw.line((other_x, other_top + other_head * 2, other_x, baseline - figure_height * 0.20), fill="#2563eb", width=stroke)
                draw.line((other_x, baseline - figure_height * 0.20, other_x - other_head, baseline), fill="#2563eb", width=stroke)
                draw.line((other_x, baseline - figure_height * 0.20, other_x + other_head, baseline), fill="#2563eb", width=stroke)
                draw.line((other_x, other_top + other_head * 2 + stroke, cx - arm_reach, arm_y), fill="#2563eb", width=stroke)
            elif action == "time":
                draw.ellipse((prop_x - prop_size, prop_y - prop_size, prop_x + prop_size, prop_y + prop_size), fill="#ffffff", outline="#2563eb", width=stroke)
                hand_angle = phase - math.pi / 2
                draw.line((prop_x, prop_y, prop_x + int(math.cos(hand_angle) * prop_size * 0.62), prop_y + int(math.sin(hand_angle) * prop_size * 0.62)), fill=accent, width=stroke)

            frames.append(image)

        animation_path = work_dir / f"stickman-animation-{scene_index:03d}.gif"
        frames[0].save(
            animation_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // 10,
            loop=0,
            optimize=True,
        )
        paths.append(animation_path)
        log(
            f"Animated scene {scene_index + 1} of {len(scenes)} "
            f"with {', '.join(action_sequence)} motions.",
            27 + round(16 * (scene_index + 1) / len(scenes)),
        )
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
    with Image.open(source) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        image = background.convert("RGB")
        if layout == "fill":
            normalized = ImageOps.fit(
                image,
                (width, height),
                method=Image.Resampling.LANCZOS,
            )
        else:
            image.thumbnail(
                (width, height),
                Image.Resampling.LANCZOS,
            )
            normalized = Image.new("RGB", (width, height), "white")
            normalized.paste(
                image,
                ((width - image.width) // 2, (height - image.height) // 2),
            )
    normalized.save(destination, "JPEG", quality=88, optimize=False)


def create_assets(
    transcript: str,
    mode: str,
    platform: str,
    voice_id: str,
    tone: str,
    layout: str,
    image_paths: list[Path],
    video_path: Path | None,
    audio_input_path: Path | None,
    work_dir: Path,
    log: Callable[[str, int], None],
    gemini_api_key: str = "",
    character_reference: bytes | None = None,
) -> tuple[Path, Path, Path]:
    transcript_path = work_dir / "transcript.txt"
    audio_path = audio_input_path or work_dir / "audio.wav"
    mp4_path = work_dir / "video.mp4"
    gif_path = work_dir / "slideshow.gif"
    transcript_path.write_text(transcript, encoding="utf-8")

    if audio_input_path is None:
        create_speech(transcript, audio_path, voice_id, tone, log)
    else:
        log("Using the uploaded audio track.", 10)
    audio_duration = ffprobe_duration(audio_path)
    log(f"Audio track duration is {audio_duration:.1f} seconds.", 22)

    output_format = FORMATS[platform]
    width = output_format["width"]
    height = output_format["height"]
    log(f"Output profile: {output_format['label']} ({width}x{height}).", 23)

    if mode == "transcript_only":
        if gemini_api_key:
            image_paths = generate_gemini_animations(
                transcript,
                work_dir,
                width,
                height,
                bool(character_reference),
                gemini_api_key,
                character_reference,
                log,
            )
        else:
            image_paths = generate_stickman_animations(
                transcript,
                work_dir,
                width,
                height,
                audio_duration,
                log,
            )

    if mode == "video" and video_path is not None:
        log("Using the uploaded video as the visual track and replacing its original audio.", 45)
        log("Encoding the video to the selected profile and narration duration.", 60)
        output_fps = 30
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
                f"setsar=1,fps={output_fps},format=yuv420p",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
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
        output_fps = 24 if mode == "audio_images" else 10
        log(f"Preparing {len(image_paths)} slideshow images for {output_format['label']}.", 45)
        normalized_images = []
        for index, image_path in enumerate(image_paths):
            progress = 46 + round(8 * index / len(image_paths))
            if mode == "transcript_only":
                normalized_images.append(image_path)
            else:
                log(f"Resizing image {index + 1} of {len(image_paths)} to {width}x{height}.", progress)
                normalized_path = work_dir / f"image-{index:03d}.jpg"
                normalize_image(image_path, normalized_path, width, height, layout)
                normalized_images.append(normalized_path)

        slideshow_images = normalized_images
        if mode == "audio_images":
            segment_count = max(
                len(normalized_images),
                min(60, math.ceil(audio_duration / 6)),
            )
            slideshow_images = [
                normalized_images[index % len(normalized_images)]
                for index in range(segment_count)
            ]
        segment_duration = audio_duration / len(slideshow_images)
        log(
            f"Timing {len(slideshow_images)} image scenes for about {segment_duration:.1f} seconds each.",
            55,
        )
        command = ["ffmpeg", "-y"]
        for image_path in slideshow_images:
            if mode == "transcript_only":
                command.extend(
                    [
                        "-stream_loop",
                        "-1",
                        "-t",
                        f"{segment_duration:.6f}",
                        "-i",
                        str(image_path),
                    ]
                )
            else:
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
        for index in range(len(slideshow_images)):
            image_filter = (
                f"{frame_filter(width, height, layout)},setsar=1,"
                f"fps={output_fps}"
            )
            if mode == "audio_images":
                zoom = (
                    "min(1+on*0.0004,1.08)"
                    if index % 2 == 0
                    else "max(1.08-on*0.0004,1)"
                )
                image_filter += (
                    f",zoompan=z='{zoom}':"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                    f"d=1:s={width}x{height}:fps={output_fps}"
                )
            filters.append(
                f"[{index}:v]{image_filter},format=yuv420p[v{index}]"
            )
        concat_inputs = "".join(f"[v{index}]" for index in range(len(slideshow_images)))
        filters.append(
            f"{concat_inputs}concat=n={len(slideshow_images)}:v=1:a=0,"
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
                f"{len(slideshow_images)}:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast" if mode == "audio_images" else "veryfast",
                "-r",
                str(output_fps),
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
        log(
            f"Encoding MP4 video (H.264, {output_fps} fps) with AAC narration using FFmpeg.",
            60,
        )
        run(command)

    log("Creating a 15-second looping GIF preview from the generated MP4.", 88)
    run(
        [
            "ffmpeg",
            "-y",
            "-t",
            "15",
            "-i",
            str(mp4_path),
            "-vf",
            "fps=4,scale=360:-1:flags=lanczos,"
            "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(gif_path),
        ]
    )
    log("MP4, GIF, and audio files are ready.", 98)
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
    audio_input_path: Path | None,
    work_dir: Path,
    gemini_api_key: str = "",
    character_reference: bytes | None = None,
) -> None:
    log_job(job_id, "Job is queued; waiting for the generation slot.", status="queued")
    with generation_lock:
        log_job(
            job_id,
            (
                (
                    "Generating AI story illustrations with Gemini and narration with "
                    f"Piper voice: {VOICE_MODELS[voice_id]['label']}."
                )
                if gemini_api_key
                else f"Generation started with Piper voice: {VOICE_MODELS[voice_id]['label']}."
                if audio_input_path is None
                else "Generation started with the uploaded audio."
            ),
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
                audio_input_path,
                work_dir,
                lambda message, progress: log_job(job_id, message, progress),
                gemini_api_key,
                character_reference,
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
    return render_template("home.html", modes=GENERATION_MODES)


@app.get("/create")
def create_page():
    mode = request.args.get("mode", "")
    if mode not in GENERATION_MODES:
        return redirect("/")
    return render_template("create.html", mode=mode, mode_info=GENERATION_MODES[mode])


@app.get("/settings")
def settings_page():
    return render_template("settings.html")


@app.get("/docs")
def documentation():
    return send_from_directory(BASE_DIR / "docs", "video-generation-flow.html")


@app.errorhandler(413)
def request_entity_too_large(_error):
    chunk_limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    if request.accept_mimetypes.best == "application/json":
        return jsonify(
            error=(
                f"One upload request exceeds the {chunk_limit_mb} MB limit. "
                "Audio and video uploads are sent in 8 MB chunks; try again or choose a smaller file."
            )
        ), 413
    return render_template(
        "upload_too_large.html",
        max_upload_mb=chunk_limit_mb,
    ), 413


def expire_upload_sessions() -> None:
    expired: list[dict] = []
    now = time.monotonic()
    with upload_sessions_lock:
        for upload_id, session in list(upload_sessions.items()):
            if now - session["updated_at"] > 12 * 60 * 60:
                expired.append(upload_sessions.pop(upload_id))
    for session in expired:
        session["temporary_path"].unlink(missing_ok=True)
        if session["complete"]:
            session["video_path"].unlink(missing_ok=True)


@app.post("/uploads")
def start_video_upload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Provide an audio or video filename and size."), 400
    filename = payload.get("filename")
    size = payload.get("size")
    if not isinstance(filename, str) or type(size) is not int or size <= 0:
        return jsonify(error="Provide a valid audio or video filename and size."), 400
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS:
        return jsonify(
            error="Use an MP4, MOV, M4V, or WebM video, or an MP3, WAV, M4A, AAC, OGG, FLAC, or OPUS audio file."
        ), 400
    if size > MAX_VIDEO_UPLOAD_BYTES:
        return jsonify(
            error=f"Video exceeds the configured {MAX_VIDEO_UPLOAD_MB} MB upload limit."
        ), 413

    expire_upload_sessions()
    upload_id = uuid.uuid4().hex
    chunk_dir = UPLOAD_DIR / ".chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = chunk_dir / f"{upload_id}.part"
    temporary_path.touch(exist_ok=False)
    with upload_sessions_lock:
        upload_sessions[upload_id] = {
            "size": size,
            "suffix": suffix,
            "received": 0,
            "updated_at": time.monotonic(),
            "temporary_path": temporary_path,
            "video_path": UPLOAD_DIR / f"{upload_id}{suffix}",
            "media_path": UPLOAD_DIR / f"{upload_id}{suffix}",
            "complete": False,
        }
    return jsonify(
        upload_id=upload_id,
        chunk_size=VIDEO_CHUNK_BYTES,
        max_upload_mb=MAX_VIDEO_UPLOAD_MB,
    ), 201


@app.post("/uploads/<upload_id>/chunks/<int:chunk_index>")
def append_video_upload_chunk(upload_id: str, chunk_index: int):
    chunk = request.get_data(cache=False)
    if not chunk:
        return jsonify(error="The upload chunk is empty."), 400
    if len(chunk) > VIDEO_CHUNK_BYTES:
        return jsonify(error="Upload chunks must not exceed 8 MB."), 413
    with upload_sessions_lock:
        session = upload_sessions.get(upload_id)
        if session is None:
            return jsonify(error="Media upload session not found or expired."), 404
        if session["complete"]:
            return jsonify(error="This media upload is already complete."), 409
        expected_index = session["received"] // VIDEO_CHUNK_BYTES
        if chunk_index != expected_index:
            return jsonify(
                error=f"Expected upload chunk {expected_index}, received {chunk_index}."
            ), 409
        if session["received"] + len(chunk) > session["size"]:
            return jsonify(error="Video upload exceeded its declared file size."), 400
        with session["temporary_path"].open("ab") as upload_file:
            upload_file.write(chunk)
        session["received"] += len(chunk)
        session["updated_at"] = time.monotonic()
        complete = session["received"] == session["size"]
        if complete:
            session["temporary_path"].replace(session["video_path"])
            session["complete"] = True
        return jsonify(
            complete=complete,
            received=session["received"],
            size=session["size"],
        )


def prepare_character_reference(image_data: bytes) -> bytes:
    with Image.open(io.BytesIO(image_data)) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        optimized = io.BytesIO()
        image.save(optimized, format="JPEG", quality=82)
    return optimized.getvalue()


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
    gemini_api_key = request.form.get("gemini_api_key", "").strip()
    uploads = request.files.getlist("images")

    if mode != "audio_images" and not transcript:
        return jsonify(error="Enter a transcript."), 400
    if mode not in {"transcript_only", "images", "video", "audio_images"}:
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
    if mode == "transcript_only":
        character_upload = request.files.get("character_image")
        if character_upload is not None and character_upload.filename:
            if Path(character_upload.filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                return jsonify(error="Use a JPG, PNG, or WebP character reference image."), 400
            try:
                character_reference = prepare_character_reference(character_upload.read())
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                return jsonify(error=f"Could not read the character reference image: {exc}"), 400
        else:
            character_reference = None
        if character_reference and not gemini_api_key:
            return jsonify(error="Enter a Gemini API key to use a character reference image."), 400
        if gemini_api_key and request.form.get("external_visual_consent") != "yes":
            return jsonify(error="Confirm the Gemini data-sharing notice before generating AI scenes."), 400
    else:
        character_reference = None
        gemini_api_key = ""

    valid_uploads = []
    image_paths = []
    video_path = None
    audio_input_path = None
    completed_upload_id = None

    if mode in {"images", "audio_images"}:
        if not uploads or all(not upload.filename for upload in uploads):
            return jsonify(error="Upload at least one image."), 400
        for upload in uploads:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                return jsonify(error="Only JPG, PNG, and WebP images are supported."), 400
            valid_uploads.append((upload, suffix))
    if mode == "audio_images":
        completed_upload_id = request.form.get("audio_upload_id")
        if completed_upload_id:
            with upload_sessions_lock:
                session = upload_sessions.get(completed_upload_id)
                if (
                    session is None
                    or not session["complete"]
                    or session["suffix"] not in ALLOWED_AUDIO_EXTENSIONS
                ):
                    return jsonify(error="Audio upload is incomplete or has expired."), 400
                audio_input_path = session["media_path"]
        else:
            upload = request.files.get("audio")
            if upload is None or not upload.filename:
                return jsonify(error="Upload an audio file."), 400
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in ALLOWED_AUDIO_EXTENSIONS:
                return jsonify(
                    error="Use an MP3, WAV, M4A, AAC, OGG, FLAC, or OPUS audio file."
                ), 400
            audio_input_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
            upload.save(audio_input_path)
            if audio_input_path.stat().st_size > MAX_VIDEO_UPLOAD_BYTES:
                audio_input_path.unlink(missing_ok=True)
                return jsonify(
                    error=f"Audio exceeds the configured {MAX_VIDEO_UPLOAD_MB} MB upload limit."
                ), 413
    elif mode == "video":
        completed_upload_id = request.form.get("video_upload_id")
        if completed_upload_id:
            with upload_sessions_lock:
                session = upload_sessions.get(completed_upload_id)
                if session is None or not session["complete"]:
                    return jsonify(error="Video upload is incomplete or has expired."), 400
                video_path = session["media_path"]
        else:
            upload = request.files.get("video")
            if upload is None or not upload.filename:
                return jsonify(error="Upload a video file."), 400
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in ALLOWED_VIDEO_EXTENSIONS:
                return jsonify(error="Use an MP4, MOV, M4V, or WebM video."), 400
            video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
            upload.save(video_path)
            if video_path.stat().st_size > MAX_VIDEO_UPLOAD_BYTES:
                video_path.unlink(missing_ok=True)
                return jsonify(
                    error=f"Video exceeds the configured {MAX_VIDEO_UPLOAD_MB} MB upload limit."
                ), 413

    for upload, suffix in valid_uploads:
        image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        upload.save(image_path)
        image_paths.append(image_path)

    job_id = uuid.uuid4().hex
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    if audio_input_path is not None:
        audio_input_path = audio_input_path.replace(
            work_dir / f"audio{audio_input_path.suffix.lower()}"
        )

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
            "visual_source": "Gemini" if gemini_api_key else "Local",
        }
    uploaded_paths = image_paths + [
        path for path in (video_path, audio_input_path) if path is not None
    ]
    total_size = sum(path.stat().st_size for path in uploaded_paths)
    if completed_upload_id:
        with upload_sessions_lock:
            upload_sessions.pop(completed_upload_id, None)
    media_description = (
        f"{len(image_paths)} images"
        if mode in {"images", "audio_images"}
        else "one video" if mode == "video" else "transcript-only stick-figure visuals"
    )
    if mode == "audio_images":
        media_description += " and uploaded audio"
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
            audio_input_path,
            work_dir,
            gemini_api_key,
            character_reference,
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
        "results.html",
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
