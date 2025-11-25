
import stable_whisper
import sys
import os

def get_output_path(source_path, new_extension):
    """Generates an output path with a new extension."""
    base_name = os.path.splitext(source_path)[0]
    return f"{base_name}{new_extension}"

def generate_captions(video_path):
    """
    Generates captions for a video file using stable-whisper.

    Args:
        video_path (str): The path to the video file.
    """
    if not os.path.exists(video_path):
        print(f"Error: File not found at '{video_path}'")
        sys.exit(1)

    print(f"Loading the Whisper model... (This may take a moment on first run)")
    try:
        model = stable_whisper.load_model('base')
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("This might be due to a network issue or missing model files.")
        print("Please ensure you have an internet connection when running for the first time.")
        sys.exit(1)

    print(f"Starting transcription for '{os.path.basename(video_path)}'. This will take some time...")
    
    try:
        # Transcribe the video
        result = model.transcribe(video_path, fp16=False) # Set fp16=True if you have a powerful GPU

        # Generate output paths
        srt_path = get_output_path(video_path, ".srt")
        vtt_path = get_output_path(video_path, ".vtt")
        tsv_path = get_output_path(video_path, ".tsv")

        # Save the results
        result.to_srt_vtt(srt_path)
        print(f"Successfully saved SRT captions to: {srt_path}")

        result.to_srt_vtt(vtt_path)
        print(f"Successfully saved VTT captions to: {vtt_path}")
        
        result.to_tsv(tsv_path)
        print(f"Successfully saved TSV data to: {tsv_path}")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python caption_generator.py <path_to_video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    generate_captions(video_file)
