import os
import glob
import subprocess
import shutil
import sys

def create_h265_video(input_dir, output_video, fps=30):
    """
    Create H.265 raw video bitstream from image sequence using FFmpeg
    :param input_dir: Input image directory
    :param output_video: Output video path (should end with .h265)
    :param fps: Frame rate (default 30)
    :return: True if successful, False otherwise
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video) or '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check FFmpeg installation
    if shutil.which("ffmpeg") is None:
        print("Error: FFmpeg is not installed or not in system PATH")
        print("Please install FFmpeg to use this function")
        return False
    
    # Get all image files
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
    
    if not image_paths:
        print(f"Error: No image files found in directory '{input_dir}'")
        return False
    
    # Sort by filename
    image_paths.sort()
    
    print("Creating H.265 raw bitstream using FFmpeg...")
    
    try:
        # Create temporary text file with image paths
        list_file = "ffmpeg_input.txt"
        with open(list_file, 'w') as f:
            for path in image_paths:
                f.write(f"file '{path}'\n")
        
        # FFmpeg command for H.265 raw bitstream encoding
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output without asking
            "-f", "concat",
            "-safe", "0",
            "-r", str(fps),
            "-i", list_file,
            "-c:v", "libx265",
            "-crf", "23",  # Quality level (0-51, lower is better)
            "-preset", "medium",  # Encoding speed/compression tradeoff
            "-pix_fmt", "yuv420p",  # Pixel format for compatibility
            "-x265-params", "log-level=error",  # Suppress unnecessary logs
            "-f", "hevc",  # Force HEVC raw bitstream output
            "-an",  # No audio
            output_video  # Output should have .h265 extension
        ]
        
        # Run FFmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        # Clean up temporary file
        os.remove(list_file)
        
        if result.returncode == 0:
            print(f"Successfully created H.265 raw bitstream: {output_video}")
            print(f"Note: This is a raw HEVC bitstream. Play with: ffplay -f hevc {output_video}")
            return True
        
        print(f"FFmpeg error: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"Video creation error: {str(e)}")
        return False

if __name__ == "__main__":
    input_directory = "C:\\Users\\Sprite\\Desktop\\hevcs\\predict"
    output_video = "C:\\Users\\Sprite\\Desktop\\hevcs\\output.h265"  # Changed to .h265 extension
    frame_rate = 60
    
    success = create_h265_video(input_directory, output_video, frame_rate)
    
    if success:
        print("H.265 raw bitstream created successfully!")
        # Open output directory on Windows
        if os.name == 'nt' and os.path.exists(output_video):
            output_dir = os.path.dirname(output_video) or '.'
            os.startfile(output_dir)
    else:
        print("Failed to create video")
        sys.exit(1)