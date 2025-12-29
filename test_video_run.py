from pathlib import Path
from pipeline.inference_service import run_combined_inference

# put any video path here (a file that exists)
video_path = Path("test_images") / "vid_1.mp4"

res = run_combined_inference(video_path)

print("RESULT:", res)
print("Output file:", Path("test_images") / res["output_filename"])
