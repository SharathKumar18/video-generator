# Video Generator

A Python and Flask app for creating YouTube and Instagram videos from a transcript:

- Transcript only: locally drawn stick-figure explainer scenes plus narration
- Transcript + image sequence: an ordered image slideshow plus narration
- Transcript + video: uploaded footage with narration replacing its original audio
- MP4, looping GIF, and WAV outputs, with live upload progress and generation logs

## Requirements

- Python 3.10+
- FFmpeg
- Piper TTS and Pillow (installed from `requirements.txt`)
- Network access on the first use of each narration voice to download its model

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

Open `http://localhost:5000`, paste a transcript, choose English or Hindi narration, a voice gender, delivery pace, generation mode, and social video format, and submit. Narration uses the transcript as written; the app does not translate it. Piper voice weights download locally on first use. Stick-figure scenes are drawn locally without an image-generation model or paid AI API.

Hindi Piper voices are licensed CC BY-NC-SA 4.0 and are for non-commercial use only. Do not select them for monetized content unless you have separate permission. The English LJSpeech and Norman voice datasets are public domain. The pace presets change speaking speed, not emotional expression.

For uploaded images and video, **Fit** is the default and preserves the full frame with padding. **Fill** crops the edges to fill the selected aspect ratio.

If port 5000 is already in use, start the app on another port:

```bash
PORT=5050 python3 app.py
```

Then open `http://localhost:5050`.

The total upload limit defaults to 250 MB. To allow larger uploads, set `MAX_UPLOAD_MB` when starting the app:

```bash
MAX_UPLOAD_MB=1024 PORT=5050 python3 app.py
```

Generated files are kept in `output/<job-id>/`. Uploaded source images are kept in `uploads/`.

See [the flow and model guide](docs/video-generation-flow.html) for the processing diagram, model/library roles, mode behavior, output profiles, and licensing notes.
