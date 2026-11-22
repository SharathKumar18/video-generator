# Video Generator

A Python and Flask app for creating YouTube and Instagram videos from a transcript:

- Transcript only: local stick-figure scenes by default, or six FLUX.1-schnell illustrations when you opt in with a Hugging Face token
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

Open `http://localhost:5000`, choose a workflow, then add its inputs. Set output format, narration language, voice, delivery pace, and framing on the separate **Global settings** screen; those preferences are saved in your browser. For **Existing audio + images**, upload an MP3, WAV, M4A, AAC, OGG, FLAC, or OPUS file and select images in the order they should appear; a transcript is not needed. The image sequence cycles with gentle zoom motion, and the video duration matches the audio. Narration uses the transcript as written; the app does not translate it. Piper voice weights download locally on first use. Transcript-only videos use local stick-figure visuals unless open-model image generation is explicitly selected.

To opt in to transcript illustrations, check **Use open-model images (FLUX.1-schnell)**, enter a Hugging Face token with Inference Providers permission, and accept the data-sharing notice. The token is used only for that job and is not saved. FLUX.1-schnell model weights are Apache 2.0, but hosted inference through Hugging Face's selected provider may incur charges; the model is gated and requires accepting Hugging Face's access conditions. Six generated images are animated and looped to fill the narration. With the box unchecked, visuals are generated locally as stick-figure scenes and no transcript is sent to an image provider. Character-reference generation is not currently supported by this hosted text-to-image model. Do not send sensitive information.

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

The blueprint uses Render's free service and sets the total audio/video upload limit to 512 MB. Audio and video chunks are sent in 8 MB requests. Free-service storage is temporary: uploaded and generated files can disappear after restarts or redeploys, and service capacity may limit the practical media size. For durable files or larger workloads, use a paid instance with persistent storage and set `DATA_DIR` to the disk's mount path. The app has no user login, so anyone with the public service URL can submit jobs; only enter a Hugging Face token on a service you trust.
