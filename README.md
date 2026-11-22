# Video Generator

A Python and Flask app for creating YouTube and Instagram videos from a transcript:

- Transcript only: animated, locally drawn stick-figure explainer scenes plus narration
- Transcript + image sequence: an ordered image slideshow plus narration
- Transcript + video: uploaded footage with narration replacing its original audio
- MP4, looping GIF, and WAV outputs, with live upload progress and generation logs

## Requirements

- Python 3.10+
- FFmpeg
- Piper TTS and Pillow (installed from `requirements.txt`)
- Network access on first use of a narration voice to download its model

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Run

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

Open `http://localhost:5000`, paste a transcript, choose English or Hindi narration, a voice gender, delivery pace, generation mode, and social video format, and submit. Narration uses the transcript as written; the app does not translate it. Piper voice weights download locally on first use. Transcript-only scenes cycle through multiple actions, with bolder stick figures, expressions, props, and background effects; they are drawn locally without an image-generation model or paid AI API. The app produces a full MP4 and a 15-second looping GIF preview.

Hindi Piper voices are licensed CC BY-NC-SA 4.0 and are for non-commercial use only. Do not select them for monetized content unless you have separate permission. The English LJSpeech and Norman voice datasets are public domain. The pace presets change speaking speed, not emotional expression.

For uploaded images and video, **Fit** is the default and preserves the full frame with padding. **Fill** crops the edges to fill the selected aspect ratio.

If port 5000 is already in use, start the app on another port:

```bash
PORT=5050 python3 app.py
```

Then open `http://localhost:5050`.

Video uploads use sequential 8 MB chunks, avoiding the per-request size limits common on hosted platforms. The total video limit defaults to 2 GB. To change it, set `MAX_VIDEO_UPLOAD_MB` when starting the app:

```bash
MAX_VIDEO_UPLOAD_MB=4096 PORT=5050 python3 app.py
```

Generated files are kept in `output/<job-id>/`. Uploaded source media is kept in `uploads/`.

## Use it online

The live app is available at **[video-generator-5m4q.onrender.com](https://video-generator-5m4q.onrender.com/)**.

The project includes a Render deployment blueprint. To create another hosted service, open [Deploy on Render](https://render.com/deploy?repo=https://github.com/SharathKumar18/video-generator), sign in, and apply the `render.yaml` blueprint.

The blueprint uses Render's free service and sets the total video upload limit to 512 MB. Video chunks are sent in 8 MB requests. Free-service storage is temporary: uploaded and generated files can disappear after restarts or redeploys, and service capacity may limit the practical video size. For durable files or larger workloads, use a paid instance with persistent storage and set `DATA_DIR` to the disk's mount path. The app has no user login, so anyone with the public service URL can submit jobs.
