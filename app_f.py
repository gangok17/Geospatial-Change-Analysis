import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from flask import Flask, render_template, request
from collections import defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =====================================================
# FLASK SETUP
# =====================================================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =====================================================
# PATHS (WINDOWS)
# =====================================================
CLASS_CSV_PATH = r"class_dict.csv"
WEIGHTS_PATH   = r"best_segformer.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# LOAD CLASS DICTIONARY
# =====================================================
class_df = pd.read_csv(CLASS_CSV_PATH)

id2rgb = {}
id2name = {}

for idx, row in class_df.iterrows():
    id2rgb[idx] = (int(row["r"]), int(row["g"]), int(row["b"]))
    id2name[idx] = row["name"] if "name" in row else f"class_{idx}"

num_classes = len(id2rgb)

# =====================================================
# BUILD LEGEND
# =====================================================
legend = [
    {"name": id2name[i], "color": f"rgb{id2rgb[i]}"}
    for i in id2rgb
]

# =====================================================
# LOAD MODEL
# =====================================================
config = SegformerConfig.from_pretrained(
    "nvidia/segformer-b3-finetuned-ade-512-512"
)
config.num_labels = num_classes

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b3-finetuned-ade-512-512",
    config=config,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

# =====================================================
# IMAGE PROCESSING
# =====================================================
def preprocess_image(image_path, size=512):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((size, size))
    tensor = TF.to_tensor(image_resized)
    tensor = TF.normalize(tensor, mean=[0.5]*3, std=[0.5]*3)
    return image, tensor.unsqueeze(0)

def predict_mask(image_path):
    original, tensor = preprocess_image(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        logits = F.interpolate(
            outputs.logits, size=(512, 512),
            mode="bilinear", align_corners=False
        )
        pred_ids = logits.argmax(dim=1)[0].cpu().numpy()

    rgb_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for cls, rgb in id2rgb.items():
        rgb_mask[pred_ids == cls] = rgb

    return original, pred_ids, rgb_mask

def overlay_mask_on_image(original_image, mask_rgb, alpha=0.5):
    mask_resized = Image.fromarray(mask_rgb).resize(
        original_image.size, Image.NEAREST
    )
    orig = np.array(original_image)
    mask = np.array(mask_resized)
    return Image.fromarray((orig*(1-alpha)+mask*alpha).astype(np.uint8))

# =====================================================
# ANALYSIS
# =====================================================
def analyze_changes(old, new):
    stats = defaultdict(int)
    for o, n in zip(old.flatten(), new.flatten()):
        if o != n:
            stats[f"{id2name[o]} → {id2name[n]}"] += 1
    return stats

def analyze_multi_year_changes(masks_dict):
    """Analyze changes between consecutive years"""
    years = sorted(masks_dict.keys())
    changes = {}
    
    for i in range(len(years)-1):
        year_from = years[i]
        year_to = years[i+1]
        changes[f"{year_from}_to_{year_to}"] = analyze_changes(
            masks_dict[year_from], masks_dict[year_to]
        )
    
    return changes

def class_percentages(mask):
    total = mask.size
    return {
        id2name[i]: round(np.sum(mask == i) / total * 100, 2)
        for i in id2name
    }

def pixels_to_hectares(px, size=10):
    return (px * size * size) / 10000

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(reports_dict, percentages_dict,
                 images_dict, masks_dict, overlay_dict):
    
    pdf_path = os.path.join(RESULT_FOLDER, "land_cover_report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    # ---------------- TITLE ----------------
    elements.append(Paragraph(
        "<b>Multi-Year Semantic Land Cover Change Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    # ---------------- IMAGES FOR EACH YEAR ----------------
    years = sorted(images_dict.keys())
    
    for year in years:
        elements.append(Paragraph(f"<b>{year} Results</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        
        img = RLImage(images_dict[year], width=2.5*inch, height=2.5*inch)
        mask = RLImage(masks_dict[year], width=2.5*inch, height=2.5*inch)
        overlay = RLImage(overlay_dict[year], width=2.5*inch, height=2.5*inch)
        
        elements.append(Table([[img, mask, overlay]]))
        elements.append(Spacer(1, 20))

    # ---------------- CHANGE TABLES FOR EACH PERIOD ----------------
    for period, report in reports_dict.items():
        year_from, year_to = period.split('_to_')
        elements.append(Paragraph(
            f"<b>Land Cover Changes ({year_from} to {year_to})</b>",
            styles["Heading2"]
        ))
        elements.append(Spacer(1, 10))

        change_table = [["Change Type", "Area (hectares)"]]
        for c, a in report:
            change_table.append([c, a])

        elements.append(Table(change_table))
        elements.append(Spacer(1, 20))

    # ---------------- PERCENTAGE TABLES FOR EACH YEAR ----------------
    for year, percentages in percentages_dict.items():
        elements.append(Paragraph(
            f"<b>Land Cover Distribution ({year})</b>",
            styles["Heading2"]
        ))
        elements.append(Spacer(1, 10))

        percent_table = [["Class", "Area (%)"]]
        for cls, pct in percentages.items():
            percent_table.append([cls, f"{pct}%"])

        elements.append(Table(percent_table))
        elements.append(Spacer(1, 20))

    # ---------------- BUILD PDF ----------------
    doc.build(elements)

    return pdf_path

# =====================================================
# ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        # Prepare dictionaries to store data for each year
        years = ['2015', '2020', '2025']
        masks = {}
        images = {}
        rgb_masks = {}
        overlays = {}
        percentages = {}
        
        # Process each year's image
        for year in years:
            file_key = f"image_{year}"
            if file_key not in request.files:
                continue
                
            file = request.files[file_key]
            if file.filename == '':
                continue
                
            # Save uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Predict mask
            img, mask, rgb_mask = predict_mask(file_path)
            masks[year] = mask
            images[year] = file_path
            rgb_masks[year] = rgb_mask
            
            # Create overlay
            overlay = overlay_mask_on_image(img, rgb_mask)
            overlays[year] = overlay
            
            # Calculate percentages
            percentages[year] = class_percentages(mask)
            
            # Save results
            Image.fromarray(rgb_mask).save(os.path.join(RESULT_FOLDER, f"mask{year}.png"))
            overlay.save(os.path.join(RESULT_FOLDER, f"overlay{year}.png"))
        
        # Analyze changes between consecutive years
        changes = analyze_multi_year_changes(masks)
        
        # Format reports for each period
        reports = {}
        for period, change_dict in changes.items():
            reports[period] = [(k, f"{pixels_to_hectares(v):.2f}") for k, v in change_dict.items()]
        
        # Generate PDF
        pdf = generate_pdf(
            reports,
            percentages,
            {year: images[year] for year in years},
            {year: f"static/results/mask{year}.png" for year in years},
            {year: f"static/results/overlay{year}.png" for year in years}
        )
        
        # Prepare results for template
        results = {
            "years": years,
            "images": {year: images[year] for year in years},
            "masks": {year: f"static/results/mask{year}.png" for year in years},
            "overlays": {year: f"static/results/overlay{year}.png" for year in years},
            "reports": reports,
            "percentages": percentages,
            "pdf": pdf
        }

    return render_template("index.html", results=results, legend=legend)


# =====================================================
# GLACIAL LAKE MODEL (SEPARATE FROM LAND COVER MODEL)
# =====================================================
import segmentation_models_pytorch as smp
import scipy.ndimage as ndi
import cv2

LAKE_MODEL_PATH = "glacial_lake_unet.pth"

lake_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

lake_model.load_state_dict(torch.load(LAKE_MODEL_PATH, map_location=device))
lake_model.eval()

# =====================================================
# ROAD MODEL (SEGFORMER)
# =====================================================
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROAD_MODEL_PATH = "road_segformer"

road_model = SegformerForSemanticSegmentation.from_pretrained(
    ROAD_MODEL_PATH,
    num_labels=2
).to(device)

road_model.eval()

road_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def resize_mask(mask, target_shape):
    """
    Resize a binary mask to target shape using nearest neighbor
    """
    return cv2.resize(
        mask.astype("uint8"),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


def detect_glacial_lake(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img) / 255.0
    H, W, _ = img_np.shape

    TILE = 512
    prob_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - TILE + 1, TILE):
            for x in range(0, W - TILE + 1, TILE):
                tile = img_np[y:y+TILE, x:x+TILE]
                tile_t = torch.tensor(tile).permute(2,0,1).unsqueeze(0).float().to(device)
                pred = torch.sigmoid(lake_model(tile_t))[0,0].cpu().numpy()
                prob_map[y:y+TILE, x:x+TILE] = pred

    # Threshold + fill
    LOW_T, HIGH_T = 0.10, 0.25
    low = (prob_map > LOW_T).astype("uint8")
    high = (prob_map > HIGH_T).astype("uint8")

    labeled, num = ndi.label(low)
    filled = np.zeros_like(low)

    for i in range(1, num + 1):
        region = (labeled == i)
        if np.any(high[region]):
            filled[region] = 1

    # NDWI water prior
    green = img_np[:,:,1]
    blue  = img_np[:,:,2]
    ndwi = (green - blue) / (green + blue + 1e-6)
    lake_mask = filled & (ndwi > 0.05)

    # Overlay
    overlay = np.array(img)
    overlay[lake_mask == 1] = [0, 0, 255]

    overlay_img = Image.fromarray(overlay)

    # Save results
    base = os.path.basename(image_path)
    mask_path = os.path.join(RESULT_FOLDER, "lake_mask_" + base)
    overlay_path = os.path.join(RESULT_FOLDER, "lake_overlay_" + base)

    Image.fromarray((lake_mask*255).astype("uint8")).save(mask_path)
    overlay_img.save(overlay_path)

    return lake_mask, overlay_path

def detect_road(image_path, threshold=0.01):   # you can tune threshold
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    inp = road_transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = road_model(pixel_values=inp)
        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits, size=(h, w),
            mode="bilinear", align_corners=False
        )

        probs = torch.softmax(logits, dim=1)

        road_prob = probs[:, 1, :, :]   # class 1 = road
        road_mask = (road_prob > threshold).squeeze().cpu().numpy().astype(np.uint8)

    return road_mask



@app.route("/glacial-lake-detection", methods=["GET", "POST"])
def glacial_lake_detection():
    result = None

    if request.method == "POST":
        file_2015 = request.files.get("image_2015")
        file_2024 = request.files.get("image_2024")

        if file_2015 and file_2024:
            # Save images
            path_2015 = os.path.join(UPLOAD_FOLDER, "2015_" + file_2015.filename)
            path_2024 = os.path.join(UPLOAD_FOLDER, "2024_" + file_2024.filename)

            file_2015.save(path_2015)
            file_2024.save(path_2024)

            # --- Run models ---
            lake_2015, _ = detect_glacial_lake(path_2015)
            lake_2024, _ = detect_glacial_lake(path_2024)

            road_2024 = detect_road(path_2024)
            # --- Ensure road mask matches lake mask ---
            if road_2024.shape != lake_2024.shape:
                road_2024 = resize_mask(road_2024, lake_2024.shape)


            # --- Area calculation ---
            area_2015 = int(np.sum(lake_2015 == 1))
            area_2024 = int(np.sum(lake_2024 == 1))
            growth_pixels = area_2024 - area_2015
            growth_percent = round((growth_pixels / max(area_2015, 1)) * 100, 2)

            # --- Ensure same shape for lake masks ---
            if lake_2015.shape != lake_2024.shape:
                lake_2015 = resize_mask(lake_2015, lake_2024.shape)

            # --- New lake expansion ---
            new_lake = (lake_2024 == 1) & (lake_2015 == 0)

            # --- Distance calculation (pixels) ---
            road_binary = (road_2024 == 1).astype(np.uint8)
            dist_map = ndi.distance_transform_edt(1 - road_binary)

            if np.any(new_lake):
                min_distance = int(dist_map[new_lake].min())
            else:
                min_distance = -1  # no expansion

            # --- Risk score ---
            growth_factor = min(growth_percent, 50) / 50
            distance_factor = max(0, (200 - min_distance) / 200) if min_distance >= 0 else 0
            risk_score = int((0.6 * growth_factor + 0.4 * distance_factor) * 100)

            if risk_score < 30:
                risk_level = "SAFE"
            elif risk_score < 60:
                risk_level = "WARNING"
            else:
                risk_level = "HIGH DANGER"

            # --- Create overlay ---
            base_img = Image.open(path_2024).convert("RGB")
            overlay = np.array(base_img)

            overlay[lake_2015 == 1] = [0, 180, 255]      # old lake
            overlay[new_lake == 1]  = [0, 0, 255]        # new expansion
            overlay[road_2024 == 1] = [255, 0, 0]        # road

            overlay_img = Image.fromarray(overlay)

            overlay_path = os.path.join(
                RESULT_FOLDER, "risk_overlay_" + file_2024.filename
            )
            overlay_img.save(overlay_path)

            result = {
                "area_2015": area_2015,
                "area_2024": area_2024,
                "growth_percent": growth_percent,
                "distance": min_distance,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "overlay": overlay_path
            }

    return render_template("glacial_lake.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)