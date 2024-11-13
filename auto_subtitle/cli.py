import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from .utils import filename, str2bool

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="large",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_ass", type=str2bool, default=False,
                        help="whether to output the .ass file along with the video files")
    parser.add_argument("--ass_only", type=str2bool, default=False,
                        help="only generate the .ass file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")
    parser.add_argument("--delay", type=float, default=0,
                        help="whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_ass: bool = args.pop("output_ass")
    ass_only: bool = args.pop("ass_only")
    language: str = args.pop("language")
    delay: float = args.pop("delay")

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    elif language != "auto":
        args["language"] = language
        
    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    subtitles = get_subtitles(
        audios, output_ass or ass_only, output_dir, lambda audio_path: model.transcribe(audio_path, word_timestamps=True, **args), delay
    )

    if ass_only:
        return

    for path, ass_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path))[0]}.mp4")
        print(f"Adding TikTok-style subtitles to {os.path.basename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        # Apply subtitles filter with styling for .ass subtitles
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

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths

def get_subtitles(audio_paths: list, output_ass: bool, output_dir: str, transcribe: callable, delay: float):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        ass_path = output_dir if output_ass else tempfile.gettempdir()
        ass_path = os.path.join(ass_path, f"{filename(path)}.ass")
        
        print(f"Generating subtitles for {filename(path)}... This might take a while.")
        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(ass_path, "w", encoding="utf-8") as ass:
            write_word_level_ass(result["segments"], delay, file=ass)

        subtitles_path[path] = ass_path

    return subtitles_path

def write_word_level_ass(segments, delay, file, window_size=5):
    # Write the ASS header
    file.write("[Script Info]\n")
    file.write("Title: Auto-generated Subtitle\n")
    file.write("ScriptType: v4.00+\n")
    file.write("Collisions: Normal\n")
    file.write("PlayDepth: 0\n\n")
    
    # Define styles: Default for regular text and Highlight for color animation only
    file.write("[V4+ Styles]\n")
    file.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    file.write("Style: Default, Montserrat, 14, &H00FFFFFF, &H00FFFFFF, &H00000000, &H64000000, 1, 0, 5, 1, 1, 5, 5, 5, 15, 1\n")  # Default style
    file.write("Style: Highlight, Montserrat, 14, &H0000FF00, &H00FFFFFF, &H00000000, &H64000000, 1, 0, 5, 1, 1, 5, 5, 5, 15, 1\n")  # Color only style

    file.write("\n[Events]\n")
    file.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    index = 1
    for segment in segments:
        words = segment['words']
        
        # Group words and display them as a static line with dynamic styling
        for i in range(0, len(words), window_size):
            group = words[i:i + window_size]
            group_text = " ".join(word['word'].upper() for word in group)  # Full group text in uppercase

            # Generate subtitle entries to animate color for each word within the static group text
            for j, word in enumerate(group):
                # Apply delay to start and end times
                start_time = format_time(word['start'] + delay)
                end_time = format_time(word['end'] + delay)

                # Build subtitle text with the current word in green only
                styled_text = []
                for k, w in enumerate(group):
                    word_upper = w['word'].upper()  # Convert to uppercase
                    if k == j:
                        # Apply the Highlight style (color only)
                        styled_text.append(f"{{\\rHighlight}}{word_upper}{{\\rDefault}}")
                    else:
                        # Regular style for other words
                        styled_text.append(word_upper)

                # Write the ASS dialogue entry for the static group, changing only the highlighted word
                file.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{' '.join(styled_text)}\n")
                index += 1

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    return f"{int(hours):01d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds*100):02d}"


if __name__ == '__main__':
    main()
