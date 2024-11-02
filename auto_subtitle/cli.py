import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from datetime import timedelta
from .utils import filename, str2bool

def format_time(seconds):
    t = timedelta(seconds=float(seconds))
    return str(t)[:-3]  # Format to HH:MM:SS.sss

def write_ass_with_word_emphasis(segments, file):
    # ASS header
    file.write("[Script Info]\n")
    file.write("Title: Auto-generated Subtitle\n")
    file.write("ScriptType: v4.00+\n")
    file.write("Collisions: Normal\n")
    file.write("PlayDepth: 0\n\n")
    file.write("[V4+ Styles]\n")
    file.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    file.write("Style: Default, Arial, 25, &H00FFFFFF, &H00FFFFFF, &H00000000, &H64000000, -1, 0, 1, 3, 0, 2, 10, 10, 30, 1\n\n")
    file.write("[Events]\n")
    file.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    # Add each word as its own dialogue line with emphasis using ASS formatting
    for segment in segments:
        if 'words' not in segment:
            print(f"No words in segment starting at {segment.get('start')}")
            continue

        for word_info in segment["words"]:
            start_time = format_time(word_info["start"])
            end_time = format_time(word_info["end"])
            word_text = word_info["word"].strip()

            # Apply emphasis by making each word bold
            file.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{{\\b1}}{word_text}{{\\b0}}\n")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str, help="paths to video files to transcribe")
    parser.add_argument("--model", default="small", choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_ass", type=str2bool, default=True, help="whether to output the .ass file along with the video files")
    parser.add_argument("--ass_only", type=str2bool, default=False, help="only generate the .ass file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", help="origin language of the video; if unset, auto-detection is used")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_ass: bool = args.pop("output_ass")
    ass_only: bool = args.pop("ass_only")
    language: str = args.pop("language")

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    elif language != "auto":
        args["language"] = language

    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    subtitles = get_subtitles(audios, output_ass or ass_only, output_dir, lambda audio_path: model.transcribe(audio_path, word_timestamps=True, **args))

    if ass_only:
        return

    for path, ass_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path))[0]}.mp4")
        print(f"Adding TikTok-style subtitles to {os.path.basename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        # Apply subtitles filter with TikTok styling
        ffmpeg.concat(
            video.filter('subtitles', ass_path),
            audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")

def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths

def get_subtitles(audio_paths: list, output_ass: bool, output_dir: str, transcribe: callable):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        ass_path = output_dir if output_ass else tempfile.gettempdir()
        ass_path = os.path.join(ass_path, f"{filename(path)}.ass")

        print(f"Generating subtitles for {filename(path)}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(ass_path, "w", encoding="utf-8") as ass:
            write_ass_with_word_emphasis(result["segments"], file=ass)

        subtitles_path[path] = ass_path

    return subtitles_path

if __name__ == '__main__':
    main()
