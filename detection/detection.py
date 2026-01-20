import logging
import os
import asyncio
import io
import tempfile
from typing import List, Optional

import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError

import cv2
import numpy as np 
from skimage.metrics import structural_similarity as compare_ssim
from ultralytics import YOLO

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Initialize the FastAPI app
logger = logging.getLogger("detection_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="House Room Classifier API",
    description="An API to classify room types for a rental website using a local Places365 model.",
    version="2.2.0",
)

# --- Configuration & Custom Labels ---

LABEL_MAPPING = {
    "bedroom": [
        "bedroom",
        "bedchamber",
        "dorm_room",
        "childs_room",
        "hotel_room",
    ],
    "other_room": [
        "dining_room",
        "dining_hall",
        "banquet_hall",
        "parlor",
        "reception",
        "television_room",
        "atrium/public",
        "waiting_room",
        "public",
        "wet_bar",
        "corridor",
        "lobby",
        "entrance_hall",
        "elevator_lobby",
        "balcony/exterior",
        "balcony/interior",
        "building_facade",
        "cottage",
        "chalet",
        "mansion",
        "beach_house",
        "manufactured_home",
        "patio",
        "courtyard",
        "yard",
        "driveway",
        "doorway/outdoor",
        "formal_garden",
        "japanese_garden",
        "outdoor",
        "exterior",
        "outdoor",
        "arch",
        "gas_station",
        "oast_house",  
        "barndoor",   
        "attic",
        "basement",
        "office",
        "utility_room",
        "storage_room",
        "closet",
        "garage/indoor",
        "dressing_room",
        "recreation_room",
        "game_room",
        "gymnasium/indoor",
        "library/indoor",
        "art_studio",
        "artists_loft",
        "bow_window/indoor",
        "indoor",
        "conference_room",
        "home_office",
        "bar",
        "burial_chamber",
        "jail_cell",
        "auditorium",
        "berth",
        "clean_room",
        "beauty_salon",
        "hospital_room",
        "chemistry_lab",
        "bathroom",
        "shower",
        "toilet",
        "laundromat",
        "lock_chamber",
##---
        "airport_terminal",
        "alcove",
        "alley",
        "amphitheater",
        "amusement_arcade",
        "amusement_park",
        "apartment_building/outdoor",
        "aquarium",
        "aqueduct",
        "arcade",
        "archive",
        "arena/hockey",
        "arena/performance",
        "arena/rodeo",
        "army_base",
        "art_gallery",
        "art_school",
        "assembly_line",
        "athletic_field/outdoor",
        "auto_factory",
        "auto_showroom",
        "bakery/shop",
        "ballroom",
        "bank_vault",
        "barn",
        "baseball_field",
        "basketball_court/indoor",
        "bazaar/indoor",
        "bazaar/outdoor",

        "beer_garden",
        "beer_hall",
        "biology_laboratory",
        "boathouse",
        "bookstore",
        "booth/indoor",
        "botanical_garden",
        "bowling_alley",
        "boxing_ring",
        "bridge",
        "bullring",
        "bus_station/indoor",
        "butchers_shop",
        "cabin/outdoor",
        "cafeteria",
        "campsite",
        "campus",
        "canal/urban",
        "candy_store",
        "castle",
        "cemetery",
        "church/indoor",
        "church/outdoor",
        "classroom",
        "closet",
        "clothing_store",
        "coffee_shop",
        "computer_room",
        "conference_center",
        "courthouse",
        "delicatessen",
        "department_store",
        "diner/outdoor",
        "discotheque",
        "downtown",
        "drugstore",
        "elevator/door",
        "elevator_shaft",
        "embassy",
        "engine_room",
        "escalator/indoor",
        "excavation",
        "fabric_store",
        "farm",
        "fastfood_restaurant",
        "field/cultivated",
        "fire_escape",
        "fire_station",
        "flea_market/indoor",
        "florist_shop/indoor",
        "food_court",
        "football_field",
        "fountain",
        "galley",
        "garage/outdoor",
        "gazebo/exterior",
        "general_store/indoor",
        "general_store/outdoor",
        "gift_shop",
        "golf_course",
        "greenhouse/indoor",
        "greenhouse/outdoor",
        "hangar/indoor",
        "hangar/outdoor",
        "harbor",
        "hardware_store",
        "home_theater",
        "hospital",
        "hotel/outdoor",
        "house",
        "hunting_lodge/outdoor",
        "ice_cream_parlor",
        "ice_skating_rink/indoor",
        "ice_skating_rink/outdoor",
        "inn/outdoor",
        "jacuzzi/indoor",
        "jewelry_shop",
        "kasbah",
        "kennel/outdoor",
        "kindergarden_classroom",
        "kitchen",
        "lecture_room",
        "legislative_chamber",
        "library/outdoor",
        "lighthouse",
        "living_room",
        "loading_dock",
        "locker_room",
        "market/indoor",
        "market/outdoor",
        "martial_arts_gym",
        "mausoleum",
        "medina",
        "mezzanine",
        "moat/water",
        "mosque/outdoor",
        "motel",
        "movie_theater/indoor",
        "museum/indoor",
        "museum/outdoor",
        "music_studio",
        "natural_history_museum",
        "nursery",
        "nursing_home",
        "office",
        "office_building",
        "office_cubicles",
        "oilrig",
        "operating_room",
        "orchard",
        "orchestra_pit",
        "palace",
        "pantry",
        "parking_garage/indoor",
        "parking_garage/outdoor",
        "parking_lot",
        "pasture",
        "pavilion",
        "pharmacy",
        "phone_booth",
        "physics_laboratory",
        "playroom",
        "plaza",
        "porch",
        "pub/indoor",
        "repair_shop",
        "residential_neighborhood",
        "restaurant",
        "restaurant_kitchen",
        "restaurant_patio",
        "roof_garden",
        "rope_bridge",
        "ruin",
        "sauna",
        "schoolhouse",
        "science_museum",
        "server_room",
        "shed",
        "shoe_shop",
        "shopfront",
        "shopping_mall/indoor",
        "ski_resort",
        "ski_slope",
        "skyscraper",
        "slum",
        "soccer_field",
        "stable",
        "stadium/baseball",
        "stadium/football",
        "stadium/soccer",
        "stage/indoor",
        "stage/outdoor",
        "staircase",
        "supermarket",
        "sushi_bar",
        "swimming_pool/indoor",
        "swimming_pool/outdoor",
        "synagogue/outdoor",
        "television_studio",
        "temple/asia",
        "throne_room",
        "ticket_booth",
        "topiary_garden",
        "tower",
        "tree_house",
        "veterinarians_office",
        "volleyball_court/outdoor",
        "wet_bar",
        "wheat_field",
        "windmill",
        "youth_hostel",
        "zen_garden",
        "interior"
    ],
}

REVERSE_LABEL_MAPPING = {
    places_label: custom_label
    for custom_label, places_labels in LABEL_MAPPING.items()
    for places_label in places_labels
}

LABELS_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
LABELS: List[str] = []

SSIM_THRESHOLD = 0.95
MAX_VIDEO_FRAMES = 6


class VideoInfo(BaseModel):
    label: str
    frame_labels: List[str]


class ImageInfoItem(BaseModel):
    index: int
    type: str


class VideoInfoItem(BaseModel):
    index: int
    type: str


class ProcessedIndices(BaseModel):
    images: List[int]
    videos: List[int]


class DetectionInfo(BaseModel):
    images_info: List[ImageInfoItem]
    video_info: List[VideoInfoItem]
    processed_indices: ProcessedIndices


class DetectionData(BaseModel):
    status: int
    message: str
    info: DetectionInfo



class DetectionResponse(BaseModel):
    data: DetectionData


def make_detection_response(
    status_code: int,
    message: str,
    *,
    images_info=None,
    video_info=None,
    processed_indices=None,
) -> JSONResponse:
    info = DetectionInfo(
        images_info=images_info or [],
        video_info=video_info or [],
        processed_indices=processed_indices
        or ProcessedIndices(images=[], videos=[]),
    )
    payload = DetectionResponse(
        data=DetectionData(status=status_code, message=message, info=info)
    )
    payload_content = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    return JSONResponse(status_code=status_code, content=payload_content)


def _collect_validation_errors(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        loc = '.'.join(str(part) for part in error.get('loc', []))
        msg = error.get('msg', 'Invalid value')
        messages.append(f"{loc}: {msg}" if loc else msg)
    return '; '.join(messages) or 'Invalid request payload'


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    message = _collect_validation_errors(exc)
    return make_detection_response(422, message)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else 'HTTP error'
    return make_detection_response(exc.status_code, detail)


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    print(f'Unhandled error: {exc}')
    return make_detection_response(500, 'Internal server error')


model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 365)
model_file = "resnet50_places365.pth.tar"

if not os.path.exists(model_file):
    print(f"Downloading model weights: {model_file} (~135 MB)... This is a one-time process.")
    url = f"http://places2.csail.mit.edu/models_places365/{model_file}"

    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            bytes_downloaded = 0
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    progress = (bytes_downloaded / total_size) * 100 if total_size > 0 else 0
                    print(
                        f"\r -> Downloading... {bytes_downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB ({progress:.2f}%)",
                        end="",
                    )
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nFailed to download model. Error: {e}")
        print("Please manually download the file from the URL above and place it in the same directory.")
        raise SystemExit(1)

try:
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise SystemExit(1)

model.eval()

human_detection_model = YOLO("yolov8n.pt")
# print("Human detection model loaded.")


try:
    response = requests.get(LABELS_URL)
    response.raise_for_status()
    LABELS = [row.split(" ")[0].split("/")[-1] for row in response.text.strip().split("\n")]
except requests.RequestException as e:
    print(f"Fatal Error: Could not download class labels: {e}")
    raise SystemExit(1)

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_image(image: Image.Image) -> str:
    # 1. Human Detection Step
    results = human_detection_model(image, verbose=False, conf=0.4) 
    
    detected_objects: List[str] = []
    logged_objects: List[str] = []
    seen_object_names: set[str] = set()
    detected_class_ids: List[int] = []
    
    if results:
        first_result = results[0]
        boxes = getattr(first_result, "boxes", None)
        if boxes is not None and boxes.cls is not None:
            for cls_id in boxes.cls.tolist():
                class_index = int(cls_id)
                detected_class_ids.append(class_index)
                class_name = first_result.names.get(class_index, str(class_index))
                detected_objects.append(class_name)
                if class_name not in seen_object_names:
                    seen_object_names.add(class_name)
                    logged_objects.append(class_name)
                    
    if logged_objects:
        logger.info("YOLO detected objects (conf>=0.4): %s", ", ".join(logged_objects))
        
    # if 0 in detected_class_ids:
    #     logger.info("Human detected in the image with high confidence.")
    #     return "not_property_pic"

    # 2. Room Classification Step (if no human was found)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    _, pred_idx = torch.max(output, 1)
    predicted_label = LABELS[pred_idx.item()]
    logger.info(f" Predicted label: {predicted_label}")

    return REVERSE_LABEL_MAPPING.get(predicted_label, "not_property_pic")


def extract_video_frames(path: str) -> List[Image.Image]:
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return frames

    previous_gray = None

    while len(frames) < MAX_VIDEO_FRAMES and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            try:
                ssim_score = compare_ssim(previous_gray, gray_frame)
                if ssim_score >= SSIM_THRESHOLD:
                    continue
            except Exception as err:
                print(f"SSIM comparison failed: {err}")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame))
        previous_gray = gray_frame

    cap.release()
    return frames


def process_video_bytes(video_bytes: bytes) -> VideoInfo:
    frame_labels: List[str] = []
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp_file.write(video_bytes)
        tmp_file.flush()
        tmp_path = tmp_file.name
    finally:
        tmp_file.close()

    try:
        frames = extract_video_frames(tmp_path)
        if not frames:
            return VideoInfo(label="no_frames_extracted", frame_labels=[])

        for frame in frames:
            try:
                frame_labels.append(predict_image(frame))
            except Exception as err:
                print(f"Frame classification error: {err}")
                frame_labels.append("classification_error")

        has_property_frame = any(label != "not_property_pic" for label in frame_labels)
        label = "property_video" if has_property_frame else "non_property_video"
        return VideoInfo(label=label, frame_labels=frame_labels)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: Optional[List[UploadFile]] = File(
        default=None,
        description="Image files (multipart form-data). Provide zero or more image files.",
    ),
    video: Optional[List[UploadFile]] = File(
        default=None,
        description="Video files (multipart form-data). Provide zero or more video files.",
    ),
) -> JSONResponse:
    image_uploads = [item for item in (image or []) if item is not None]
    video_uploads = [item for item in (video or []) if item is not None]

    if not image_uploads and not video_uploads:
        logger.warning("Detect called without media uploads")
        return make_detection_response(
            400,
            "Provide at least one image or video file.",
            images_info=[],
            video_info=[],
            processed_indices=ProcessedIndices(images=[], videos=[]),
        )

    batches = [(upload, "image") for upload in image_uploads]
    batches += [(upload, "video") for upload in video_uploads]

    image_items: List[ImageInfoItem] = []
    processed_images: List[int] = []

    video_items: List[VideoInfoItem] = []
    processed_videos: List[int] = []

    image_index = 0
    video_index = 0

    has_success = False
    has_semantic_failure = False
    has_internal_error = False
    has_unsupported = False

    for index, (upload, media_kind) in enumerate(batches):
        try:
            file_bytes = await upload.read()
        except Exception as err:
            logger.exception("Upload read failure at slot %s", index)
            has_internal_error = True
            continue

        content_type = (upload.content_type or "").lower()
        filename = (upload.filename or "").lower()

        if media_kind == "video":
            is_video = True
        elif media_kind == "image":
            is_video = False
        else:
            is_video = any(
                token in content_type for token in ("video/", "mp4", "quicktime", "x-msvideo", "x-matroska")
            ) or filename.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"))

        if not file_bytes:
            logger.warning("Empty file received at slot %s", index)
            if is_video:
                video_items.append(VideoInfoItem(index=video_index, type="empty_file"))
                video_index += 1
                has_semantic_failure = True
            else:
                image_items.append(ImageInfoItem(index=image_index, type="empty_file"))
                image_index += 1
                has_semantic_failure = True
            try:
                await upload.close()
            except Exception:
                pass
            continue

        if not is_video:
            try:
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                label = predict_image(image)
            except UnidentifiedImageError:
                logger.warning("Unsupported media at slot %s", index)
                label = "unsupported_media"
                has_unsupported = True
            except Exception as err:
                logger.exception("Image classification error at slot %s", index)
                label = "classification_error"
                has_internal_error = True

            image_items.append(ImageInfoItem(index=image_index, type=label))
            if label not in {"classification_error", "empty_file", "unsupported_media"}:
                if label != "not_property_pic":
                    processed_images.append(image_index)
                has_success = True
            else:
                if label == "unsupported_media":
                    has_unsupported = True
                elif label == "classification_error":
                    has_internal_error = True
                else:
                    has_semantic_failure = True
            image_index += 1
        else:
            info = VideoInfo(label="classification_error", frame_labels=[])
            try:
                info = await asyncio.to_thread(process_video_bytes, file_bytes)
            except Exception as err:
                logger.exception("Video classification error at slot %s", index)
                info = VideoInfo(label="classification_error", frame_labels=[])
                has_internal_error = True
            video_items.append(VideoInfoItem(index=video_index, type=info.label))
            if info.label == "property_video":
                processed_videos.append(video_index)
                has_success = True
            elif info.label == "non_property_video":
                has_success = True
            elif info.label == "empty_file":
                has_semantic_failure = True
            elif info.label == "download_failed":
                has_semantic_failure = True
            elif info.label == "no_frames_extracted":
                has_semantic_failure = True
            else:
                has_internal_error = True
            video_index += 1

        try:
            await upload.close()
        except Exception:
            pass

    processed_indices = ProcessedIndices(images=processed_images, videos=processed_videos)

    status_code = 200
    message = "Success!"

    if has_success:
        if has_semantic_failure or has_internal_error or has_unsupported:
            status_code = 207
            message = "Partial success."
    else:
        if has_internal_error:
            status_code = 500
            message = "Internal server error."
        elif has_unsupported and not has_semantic_failure:
            status_code = 415
            message = "Unsupported media type."
        else:
            status_code = 422
            message = "Could not process uploaded media."

    logger.info(
        "Detect completed status=%s successes=%s semantic_failures=%s internal_errors=%s unsupported=%s",
        status_code,
        has_success,
        has_semantic_failure,
        has_internal_error,
        has_unsupported,
    )

    return make_detection_response(
        status_code,
        message,
        images_info=image_items,
        video_info=video_items,
        processed_indices=processed_indices,
    )
