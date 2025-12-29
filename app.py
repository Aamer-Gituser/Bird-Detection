# app.py
from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)

from werkzeug.utils import secure_filename

from pipeline.inference_service import (
    run_combined_inference,
    IMAGE_EXTS,
    VIDEO_EXTS,
)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "test_images"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = IMAGE_EXTS | VIDEO_EXTS

app = Flask(__name__)


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------
# Media serving (input + output files)
# -------------------------------------------------------

@app.route("/media/<path:filename>")
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if not allowed_file(file.filename):
            return redirect(request.url)

        # Secure + unique filename
        safe_name = secure_filename(file.filename)
        unique_name = f"{uuid4().hex}_{safe_name}"
        save_path = MEDIA_DIR / unique_name

        # Save uploaded file
        file.save(str(save_path))

        # -----------------------------
        # Run inference
        # -----------------------------
        result = run_combined_inference(save_path)

        # -----------------------------
        # URLs for frontend
        # -----------------------------
        result["input_url"] = url_for(
            "media", filename=result["input_filename"]
        )

        if result.get("output_filename"):
            result["output_url"] = url_for(
                "media", filename=result["output_filename"]
            )
        else:
            result["output_url"] = None

        return render_template("result.html", result=result)

    # GET request
    return render_template("index.html")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
