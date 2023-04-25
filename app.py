import json
import subprocess
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import DEFAULT_MODEL, get_model
from huggingface_hub import hf_hub_download, list_repo_files

from so_vits_svc_fork.hparams import HParams
from so_vits_svc_fork.inference.core import Svc


###################################################################
# REPLACE THESE VALUES TO CHANGE THE MODEL REPO/CKPT NAME/SETTINGS
###################################################################
# The Hugging Face Hub repo ID
repo_id = "dog/kanye"
# If None, Uses latest ckpt in the repo
ckpt_name = None
# If None, Uses "kmeans.pt" if it exists in the repo
cluster_model_name = None
# Set the default f0 type to use - use the one it was trained on.
# The default for so-vits-svc-fork is "dio".
# Options: "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
default_f0_method = "crepe"
# The default ratio of cluster inference to SVC inference.
# If cluster_model_name is not found in the repo, this is set to 0.
default_cluster_infer_ratio = 0.5
###################################################################

# Figure out the latest generator by taking highest value one.
# Ex. if the repo has: G_0.pth, G_100.pth, G_200.pth, we'd use G_200.pth
if ckpt_name is None:
    latest_id = sorted(
        [
            int(Path(x).stem.split("_")[1])
            for x in list_repo_files(repo_id)
            if x.startswith("G_") and x.endswith(".pth")
        ]
    )[-1]
    ckpt_name = f"G_{latest_id}.pth"

cluster_model_name = cluster_model_name or "kmeans.pt"
if cluster_model_name in list_repo_files(repo_id):
    print(f"Found Cluster model - Downloading {cluster_model_name} from {repo_id}")
    cluster_model_path = hf_hub_download(repo_id, cluster_model_name)
else:
    print(f"Could not find {cluster_model_name} in {repo_id}. Using None")
    cluster_model_path = None
default_cluster_infer_ratio = default_cluster_infer_ratio if cluster_model_path else 0

generator_path = hf_hub_download(repo_id, ckpt_name)
config_path = hf_hub_download(repo_id, "config.json")
hparams = HParams(**json.loads(Path(config_path).read_text()))
speakers = list(hparams.spk.keys())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Svc(net_g_path=generator_path, config_path=config_path, device=device, cluster_model_path=cluster_model_path)
demucs_model = get_model(DEFAULT_MODEL)


def extract_vocal_demucs(model, filename, sr=44100, device=None, shifts=1, split=True, overlap=0.25, jobs=0):
    wav, sr = librosa.load(filename, mono=False, sr=sr)
    wav = torch.tensor(wav)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(
        model, wav[None], device=device, shifts=shifts, split=split, overlap=overlap, progress=True, num_workers=jobs
    )[0]
    sources = sources * ref.std() + ref.mean()
    # We take just the vocals stem. I know the vocals for this model are at index -1
    # If using different model, check model.sources.index('vocals')
    vocal_wav = sources[-1]
    # I did this because its the same normalization the so-vits model required
    vocal_wav = vocal_wav / max(1.01 * vocal_wav.abs().max(), 1)
    vocal_wav = vocal_wav.numpy()
    vocal_wav = librosa.to_mono(vocal_wav)
    vocal_wav = vocal_wav.T
    instrumental_wav = sources[:-1].sum(0).numpy().T
    return vocal_wav, instrumental_wav


def download_youtube_clip(
    video_identifier,
    start_time,
    end_time,
    output_filename,
    num_attempts=5,
    url_base="https://www.youtube.com/watch?v=",
    quiet=False,
    force=False,
):
    output_path = Path(output_filename)
    if output_path.exists():
        if not force:
            return output_path
        else:
            output_path.unlink()

    quiet = "--quiet --no-warnings" if quiet else ""
    command = f"""
        yt-dlp {quiet} -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"  # noqa: E501
    """.strip()

    attempts = 0
    while True:
        try:
            _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            attempts += 1
            if attempts == num_attempts:
                return None
        else:
            break

    if output_path.exists():
        return output_path
    else:
        return None


def predict(
    speaker,
    audio,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: str = "crepe",
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
):
    audio, _ = librosa.load(audio, sr=model.target_sample)
    audio = model.infer_silence(
        audio.astype(np.float32),
        speaker=speaker,
        transpose=transpose,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        f0_method=f0_method,
        db_thresh=db_thresh,
        pad_seconds=pad_seconds,
        chunk_seconds=chunk_seconds,
        absolute_thresh=absolute_thresh,
    )
    return model.target_sample, audio


def predict_song_from_yt(
    ytid_or_url,
    start,
    end,
    speaker=speakers[0],
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: str = "dio",
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
):
    original_track_filepath = download_youtube_clip(
        ytid_or_url,
        start,
        end,
        "track.wav",
        force=True,
        url_base="" if ytid_or_url.startswith("http") else "https://www.youtube.com/watch?v=",
    )
    vox_wav, inst_wav = extract_vocal_demucs(demucs_model, original_track_filepath)
    if transpose != 0:
        inst_wav = librosa.effects.pitch_shift(inst_wav.T, sr=model.target_sample, n_steps=transpose).T
    cloned_vox = model.infer_silence(
        vox_wav.astype(np.float32),
        speaker=speaker,
        transpose=transpose,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        f0_method=f0_method,
        db_thresh=db_thresh,
        pad_seconds=pad_seconds,
        chunk_seconds=chunk_seconds,
        absolute_thresh=absolute_thresh,
    )
    full_song = inst_wav + np.expand_dims(cloned_vox, 1)
    return (model.target_sample, full_song), (model.target_sample, cloned_vox)


SPACE_ID = "nateraw/voice-cloning"
description = f"""
# Attention - This Space may be slow in the shared UI if there is a long queue. To speed it up, you can duplicate and use it with a paid private T4 GPU.

<center><a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></center>

## This app uses models trained with [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork) to clone a voice. Model currently being used is https://hf.co/{repo_id}.

#### To change the model being served, duplicate the space and update the `repo_id`/other settings in `app.py`.

#### Train Your Own: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nateraw/voice-cloning/blob/main/training_so_vits_svc_fork.ipynb)
""".strip()

article = """
<p style='text-align: center'>
    <a href='https://github.com/voicepaw/so-vits-svc-fork' target='_blank'>Github Repo</a>
</p>
""".strip()


interface_mic = gr.Interface(
    predict,
    inputs=[
        gr.Dropdown(speakers, value=speakers[0], label="Target Speaker"),
        gr.Audio(type="filepath", source="microphone", label="Source Audio"),
        gr.Slider(-12, 12, value=0, step=1, label="Transpose (Semitones)"),
        gr.Checkbox(False, label="Auto Predict F0"),
        gr.Slider(0.0, 1.0, value=default_cluster_infer_ratio, step=0.1, label="cluster infer ratio"),
        gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="noise scale"),
        gr.Dropdown(
            choices=["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
            value=default_f0_method,
            label="f0 method",
        ),
    ],
    outputs="audio",
    title="Voice Cloning",
    description=description,
    article=article,
)
interface_file = gr.Interface(
    predict,
    inputs=[
        gr.Dropdown(speakers, value=speakers[0], label="Target Speaker"),
        gr.Audio(type="filepath", source="upload", label="Source Audio"),
        gr.Slider(-12, 12, value=0, step=1, label="Transpose (Semitones)"),
        gr.Checkbox(False, label="Auto Predict F0"),
        gr.Slider(0.0, 1.0, value=default_cluster_infer_ratio, step=0.1, label="cluster infer ratio"),
        gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="noise scale"),
        gr.Dropdown(
            choices=["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
            value=default_f0_method,
            label="f0 method",
        ),
    ],
    outputs="audio",
    title="Voice Cloning",
    description=description,
    article=article,
)
interface_yt = gr.Interface(
    predict_song_from_yt,
    inputs=[
        gr.Textbox(
            label="YouTube URL or ID", info="A YouTube URL (or ID) to a song on YouTube you want to clone from"
        ),
        gr.Number(value=0, label="Start Time (seconds)"),
        gr.Number(value=15, label="End Time (seconds)"),
        gr.Dropdown(speakers, value=speakers[0], label="Target Speaker"),
        gr.Slider(-12, 12, value=0, step=1, label="Transpose (Semitones)"),
        gr.Checkbox(False, label="Auto Predict F0"),
        gr.Slider(0.0, 1.0, value=default_cluster_infer_ratio, step=0.1, label="cluster infer ratio"),
        gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="noise scale"),
        gr.Dropdown(
            choices=["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
            value=default_f0_method,
            label="f0 method",
        ),
    ],
    outputs=["audio", "audio"],
    title="Voice Cloning",
    description=description,
    article=article,
    examples=[
        ["COz9lDCFHjw", 75, 90, speakers[0], 0, False, default_cluster_infer_ratio, 0.4, default_f0_method],
        ["dQw4w9WgXcQ", 21, 35, speakers[0], 0, False, default_cluster_infer_ratio, 0.4, default_f0_method],
        ["Wvm5GuDfAas", 15, 30, speakers[0], 0, False, default_cluster_infer_ratio, 0.4, default_f0_method],
    ],
)
interface = gr.TabbedInterface(
    [interface_mic, interface_file, interface_yt],
    ["Clone From Mic", "Clone From File", "Clone Song From YouTube"],
)


if __name__ == "__main__":
    interface.launch()
