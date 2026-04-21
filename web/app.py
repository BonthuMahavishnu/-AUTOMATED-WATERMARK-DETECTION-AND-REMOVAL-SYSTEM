from flask import Flask, render_template, request
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from model.inference import remove_watermark

app = Flask(__name__)

UPLOAD_DIR = BASE_DIR / "web/static/uploads"
OUTPUT_DIR = BASE_DIR / "web/static/outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            input_path = UPLOAD_DIR / file.filename
            output_path = OUTPUT_DIR / f"cleaned_{file.filename}"

            file.save(input_path)
            remove_watermark(input_path, output_path)

            result_image = f"static/outputs/cleaned_{file.filename}"

    return render_template("index.html", result=result_image)

if __name__ == "__main__":
    app.run(debug=True)