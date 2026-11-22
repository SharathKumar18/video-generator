# Video Generator

A Python and Flask app for creating YouTube and Instagram videos from a transcript:

- Transcript only: local stick-figure scenes by default, or six Gemini illustrations when you provide a Gemini API key; a character reference produces eight scenes
- Transcript + image sequence: an ordered image slideshow plus narration
- Existing audio + images: cycle ordered images with gentle zoom motion to match an uploaded audio track
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

Open `http://localhost:5000`, choose a workflow, then add its inputs. Set output format, narration language, voice, delivery pace, and framing on the separate **Global settings** screen; those preferences are saved in your browser. For **Existing audio + images**, upload an MP3, WAV, M4A, AAC, OGG, FLAC, or OPUS file and select images in the order they should appear; a transcript is not needed. The image sequence cycles with gentle zoom motion, and the video duration matches the audio. Narration uses the transcript as written; the app does not translate it. Piper voice weights download locally on first use. Without a Gemini key, transcript-only videos use the built-in local illustrations.

For AI-generated transcript scenes, enter your Google Gemini API key on the transcript input screen. The key is used only for that job and is not saved. Gemini creates six story illustrations, or eight when you provide a character reference image; the app animates and repeats those scenes to fill the narration. Without a key, the app uses local stick-figure scenes. A character reference requires a Gemini key. The transcript and any character reference image are sent to Google only when you provide a key and accept the disclosure. Gemini API use requires the user to be 18 or older. On unpaid API tiers, Google may use submitted inputs to improve services and human reviewers may process them; paid-tier terms differ. Charges may apply. See [Gemini API terms](https://aistudio.google.com/docs/terms) and [image generation models](https://aistudio.google.com/docs/image-generation). Do not enter a key or transcript containing sensitive information unless you have checked your API tier's terms.

Hindi Piper voices are licensed CC BY-NC-SA 4.0 and are for non-commercial use only. Do not select them for monetized content unless you have separate permission. The English LJSpeech and Norman voice datasets are public domain. The pace presets change speaking speed, not emotional expression.

For uploaded images, audio-led slideshows, and video, **Fit** is the default and preserves the full frame with padding. **Fill** crops the edges to fill the selected aspect ratio. Image resizing is done locally with Pillow; FFmpeg is used for final video encoding.

If port 5000 is already in use, start the app on another port:

```bash
PORT=5050 python3 app.py
```

Then open `http://localhost:5050`.

Audio and video uploads use sequential 8 MB chunks, avoiding the per-request size limits common on hosted platforms. The total media limit defaults to 2 GB. To change it, set `MAX_VIDEO_UPLOAD_MB` when starting the app:

```bash
MAX_VIDEO_UPLOAD_MB=4096 PORT=5050 python3 app.py
```

Generated files are kept in `output/<job-id>/`. Uploaded source media is kept in `uploads/`.

## Use it online

The live app is available at **[video-generator-5m4q.onrender.com](https://video-generator-5m4q.onrender.com/)**.

The project includes a Render deployment blueprint. To create another hosted service, open [Deploy on Render](https://render.com/deploy?repo=https://github.com/SharathKumar18/video-generator), sign in, and apply the `render.yaml` blueprint.

The blueprint uses Render's free service and sets the total audio/video upload limit to 512 MB. Audio and video chunks are sent in 8 MB requests. Free-service storage is temporary: uploaded and generated files can disappear after restarts or redeploys, and service capacity may limit the practical media size. For durable files or larger workloads, use a paid instance with persistent storage and set `DATA_DIR` to the disk's mount path. The app has no user login, so anyone with the public service URL can submit jobs; only enter a Gemini key on a service you trust.
