import os
import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
from flask import Flask, render_template, request, send_from_directory, jsonify, make_response, redirect, url_for
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO
from openpyxl.styles import Border, Side
from PIL import Image, ImageOps
from pytesseract import Output

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# 1. PATH CONFIGURATION (Auto-Detect)
# ==========================================
if os.name == 'nt': # Windows (Your Laptop)
    POPPLER_PATH = r"C:\Users\Paladi Gurunadh\OneDrive\Desktop\teserract\poppler\poppler-25.12.0\Library\bin"
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    if not os.path.exists(POPPLER_PATH):
        print(f"\n⚠️ WARNING: Poppler path not found at: {POPPLER_PATH}\n")
else: # Linux (Cloud/Render)
    POPPLER_PATH = None 
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

extraction_history = []
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# 2. UNIVERSAL FILE HANDLER
# ==========================================
def save_and_standardize(file):
    """Converts PDF or Image to a clean RGB PNG"""
    original_name = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], original_name)
    file.save(filepath)
    
    final_image = None
    
    # Handle PDF
    if original_name.lower().endswith(".pdf"):
        try:
            pages = convert_from_path(filepath, dpi=300, first_page=1, last_page=1, poppler_path=POPPLER_PATH)
            final_image = pages[0]
        except Exception as e:
            raise Exception(f"Poppler Error: {str(e)}")
            
    # Handle Images
    else:
        try:
            img = Image.open(filepath)
            img = ImageOps.exif_transpose(img)
            final_image = img.convert("RGB")
        except Exception as e:
            raise Exception(f"Image Error: {str(e)}")

    # Save as Clean PNG
    clean_filename = original_name.rsplit('.', 1)[0] + "_clean.png"
    clean_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_filename)
    
    if final_image:
        final_image.save(clean_path, "PNG")
        return clean_filename
    
    raise Exception("File Conversion Failed")

def calculate_confidence(img):
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        confidences = [int(c) for c in data['conf'] if c != '-1']
        return round(sum(confidences) / len(confidences), 2) if confidences else 0
    except:
        return 0

# ==========================================
# 3. AI ENGINES
# ==========================================

def engine_visionscan(img):
    """👑 VISIONSCAN ADVANCED: Table & Text Optimizer"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check for Grid/Table
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None and len(lines) > 5:
        # Table Mode: Grid Removal
        img_r = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detect_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        detect_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        
        mask = cv2.addWeighted(detect_h, 0.5, detect_v, 0.5, 0.0)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        
        return cv2.inpaint(img_r, mask, 3, cv2.INPAINT_TELEA)
    else:
        # Document Mode: Contrast & Sharpen
        img_r = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_r)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(enhanced, -1, kernel)

# Standard Engines
def engine_raw(img): return img
def engine_grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def engine_binary(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def engine_blur(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)
def engine_inverted(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(gray)

# ==========================================
# 4. FLASK ROUTES
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():
    filename = None
    error_msg = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            if not allowed_file(file.filename):
                error_msg = "❌ Invalid Format!"
                return render_template("demo.html", error=error_msg)
            try:
                filename = save_and_standardize(file)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
        else:
            error_msg = "⚠️ No file selected."
    
    return render_template("demo.html", filename=filename, error=error_msg)

@app.route("/benchmark", methods=["GET", "POST"])
def benchmark():
    results = []
    filename = None
    error_msg = None
    
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            try:
                # Load Image
                filename = save_and_standardize(file)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                pil_img = Image.open(filepath).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Define Engines
                engines = [
                    ("VisionScan Advanced (Ours)", engine_visionscan, "👑", "success"),
                    ("Standard Raw", engine_raw, "📄", "secondary"),
                    ("Grayscale", engine_grayscale, "⚫", "secondary"),
                    ("Binary Threshold", engine_binary, "wb_contrast", "secondary"),
                    ("Gaussian Blur", engine_blur, "blur_on", "secondary"),
                    ("Inverted Colors", engine_inverted, "invert_colors", "secondary")
                ]
                
                # Run Comparison
                for name, func, icon, css_class in engines:
                    try:
                        processed_img = func(img.copy())
                        conf = calculate_confidence(processed_img)
                        config = r'--psm 6' if "VisionScan" in name else r'--psm 3'
                        text = pytesseract.image_to_string(processed_img, config=config)
                        results.append({"name": name, "conf": conf, "text": text, "class": css_class, "icon": icon})
                    except Exception as inner_e:
                        results.append({"name": name, "conf": 0, "text": "Error", "class": "danger", "icon": "error"})
                        
            except Exception as e:
                error_msg = str(e)
            
    # CRITICAL FIX: Ensure this points to 'benchmark.html'
    return render_template("benchmark.html", results=results, filename=filename, error=error_msg)

@app.route("/crop_ocr", methods=["POST"])
def crop_ocr():
    data = request.json
    filename = data['filename']
    x, y, w, h = int(data['x']), int(data['y']), int(data['w']), int(data['h'])
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    pil_img = Image.open(filepath).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    crop_img = img[y:y+h, x:x+w]
    processed = engine_visionscan(crop_img)
    text = pytesseract.image_to_string(processed, config=r'--oem 3 --psm 6 -c preserve_interword_spaces=1').strip()
    if text: extraction_history.insert(0, text)
    return jsonify({"text": text, "history": extraction_history})

@app.route("/export_excel", methods=["POST"])
def export_excel():
    if not extraction_history: return "No data", 400
    rows = []
    for line in extraction_history[0].split('\n'):
        if line.strip():
            cols = [c.strip() for c in re.split(r'\s{2,}', line.replace('|',' ')) if c.strip()]
            rows.append(cols)
    df = pd.DataFrame(rows)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, header=False)
    output.seek(0)
    response = make_response(output.read())
    response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    response.headers['Content-Disposition'] = 'attachment; filename=table.xlsx'
    return response

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    if not extraction_history: return "No data", 400
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    y = 800
    p.setFont("Courier", 10)
    for line in extraction_history[0].split('\n'):
        if y < 50: p.showPage(); p.setFont("Courier", 10); y = 800
        p.drawString(50, y, line)
        y -= 12
    p.save()
    buffer.seek(0)
    response = make_response(buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=extracted.pdf'
    return response

@app.route("/reset_app")
def reset_app(): 
    global extraction_history
    extraction_history = []
    return redirect(url_for('index'))

@app.route("/analysis")
def analysis(): return render_template("analysis.html", count=len(extraction_history))
@app.route("/survey")
def survey(): return render_template("survey.html")
@app.route("/vlm")
def vlm(): return render_template("vlm.html")
@app.route("/recommendations")
def recommendations(): return render_template("recommendations.html")
@app.route('/uploads/<filename>')
def uploaded_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)