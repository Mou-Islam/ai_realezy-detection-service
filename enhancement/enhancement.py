import asyncio
import base64
import logging
import shutil
import sys
import types
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional
from urllib.request import urlopen

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

try:  # pragma: no cover - optional dependency bridge
    import torchvision.transforms.functional as _tv_funct
except ImportError:  # pragma: no cover - optional
    _tv_funct = None
else:  # pragma: no cover - compatibility shim
    if hasattr(_tv_funct, "rgb_to_grayscale") and "torchvision.transforms.functional_tensor" not in sys.modules:
        module = types.ModuleType("torchvision.transforms.functional_tensor")
        module.rgb_to_grayscale = _tv_funct.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = module


try:  # pragma: no cover - optional dependency
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    from realesrgan import RealESRGANer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency missing
    RRDBNet = None
    RealESRGANer = None


logger = logging.getLogger("enhancement_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/enhancement", tags=["enhancement"])

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
MODEL_FILENAME = "RealESRGAN_x4plus.pth"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

UPSAMPLER: Optional[Any] = None
UPSAMPLER_LOCK = Lock()
UPSAMPLER_ERROR: Optional[str] = None

DEFAULT_SCALE = 2.0
DEFAULT_DENOISE = 7
DEFAULT_SHARPEN = 1.0
DEFAULT_CONTRAST = 2.0
DEFAULT_USE_REALESRGAN = True


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def restore_alpha(source: Image.Image, target: Image.Image) -> Image.Image:
    if source.mode in ("RGBA", "LA"):
        alpha = source.split()[-1].resize(target.size, resample=Image.LANCZOS)
        target = target.convert("RGBA")
        target.putalpha(alpha)
    return target


def _safe_filename(name: Optional[str], index: int) -> str:
    base = (Path(name).name if name else "").strip()
    if not base:
        return f"upload_{index + 1}.png"
    return base



def _ensure_model_weights() -> Optional[Path]:
    path = WEIGHTS_DIR / MODEL_FILENAME
    if path.exists():
        return path
    if RealESRGANer is None:
        return None

    try:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        with urlopen(MODEL_URL) as response, open(path, "wb") as dest:
            shutil.copyfileobj(response, dest)
        logger.info("Downloaded Real-ESRGAN weights to %s", path)
        return path
    except Exception as exc:  # pragma: no cover - network issue
        logger.warning("Failed to download Real-ESRGAN weights: %s", exc)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        return None


def _init_realesrgan() -> Optional[Any]:
    global UPSAMPLER, UPSAMPLER_ERROR

    if RealESRGANer is None or RRDBNet is None:
        UPSAMPLER_ERROR = "Real-ESRGAN is not installed."
        return None

    if UPSAMPLER is not None:
        return UPSAMPLER

    with UPSAMPLER_LOCK:
        if UPSAMPLER is not None:
            return UPSAMPLER

        weights_path = _ensure_model_weights()
        if weights_path is None:
            UPSAMPLER_ERROR = "Real-ESRGAN weights unavailable."
            return None

        try:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            try:
                import torch

                use_half = torch.cuda.is_available()
            except Exception:  # pragma: no cover - torch import failure
                use_half = False
                tile_size = 0
            else:
                tile_size = 0 if use_half else 256

            UPSAMPLER = RealESRGANer(
                scale=4,
                model_path=str(weights_path),
                model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=use_half,
            )
            UPSAMPLER_ERROR = None
            logger.info("Real-ESRGAN initialized with %s", weights_path)
        except Exception as exc:  # pragma: no cover - init failure
            UPSAMPLER = None
            UPSAMPLER_ERROR = str(exc)
            logger.warning("Real-ESRGAN initialization failed: %s", exc)

        return UPSAMPLER


def enhance_image(
    pil_img: Image.Image,
    scale: float = DEFAULT_SCALE,
    denoise_strength: int = DEFAULT_DENOISE,
    sharpen_amount: float = DEFAULT_SHARPEN,
    contrast_clip_limit: float = DEFAULT_CONTRAST,
) -> Image.Image:
    img = pil_img.convert("RGB")
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    h = clamp(float(denoise_strength), 0.0, 20.0)
    if h > 0:
        np_img = cv2.fastNlMeansDenoisingColored(np_img, None, h, h, 7, 21)

    lab = cv2.cvtColor(np_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = clamp(float(contrast_clip_limit), 1.0, 4.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    np_img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    amount = clamp(float(sharpen_amount), 0.0, 3.0)
    if amount > 0:
        blur = cv2.GaussianBlur(np_img, (0, 0), sigmaX=1.2, sigmaY=1.2)
        np_img = cv2.addWeighted(np_img, 1 + amount, blur, -amount, 0)

    s = clamp(float(scale), 1.0, 4.0)
    if s != 1.0:
        h0, w0 = np_img.shape[:2]
        new_w, new_h = int(w0 * s), int(h0 * s)
        np_img = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    out_pil = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return restore_alpha(pil_img, out_pil)


def enhance_with_realesrgan(pil_img: Image.Image, scale: float = DEFAULT_SCALE) -> Image.Image:
    upsampler = _init_realesrgan()
    if upsampler is None:
        raise RuntimeError(UPSAMPLER_ERROR or "Real-ESRGAN unavailable")

    rgb = pil_img.convert("RGB")
    np_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    out, _ = upsampler.enhance(np_img, outscale=scale)
    enhanced = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    return restore_alpha(pil_img, enhanced)


@router.on_event("startup")
async def warm_realesrgan() -> None:  # pragma: no cover - runs once
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _init_realesrgan)


@router.post("/enhance")
async def enhance_endpoint(
    file: List[UploadFile] = File(
        ...,
        description="Image files (png/jpg/webp); provide one or more files.",
    ),
):
    uploads = [item for item in (file or []) if item is not None]
    if not uploads:
        raise HTTPException(status_code=400, detail="Provide at least one image file.")

    results: List[dict[str, Any]] = []
    for idx, upload in enumerate(uploads):
        meta: dict[str, Any] = {
            "index": idx,
            "filename": _safe_filename(upload.filename, idx),
            "status": "error",
            "detail": None,
            "error_code": 400,
            "backend": None,
            "warning": None,
            "content_type": None,
            "size": None,
            "output_filename": None,
        }
        payload: Optional[bytes] = None
        raw: Optional[bytes] = None
        pil_img: Optional[Image.Image] = None

        try:
            content_type = (upload.content_type or "").lower()
            if content_type and not any(token in content_type for token in ("image", "octet-stream")):
                logger.warning("Unsupported content type supplied at slot %s: %s", idx, content_type)
                meta["detail"] = "Unsupported content type."
                meta["error_code"] = 415

            if meta["detail"] is None:
                try:
                    raw = await upload.read()
                except Exception as exc:
                    logger.exception("Unhandled error while reading upload at slot %s", idx)
                    meta["detail"] = f"Failed to read image: {exc}"
                    meta["error_code"] = 400

            if meta["detail"] is None and not raw:
                logger.warning("Empty payload received for enhancement request at slot %s", idx)
                meta["detail"] = "Empty file uploaded."
                meta["error_code"] = 400

            if meta["detail"] is None:
                try:
                    pil_img = Image.open(BytesIO(raw))
                    pil_img.load()
                except UnidentifiedImageError as exc:
                    logger.warning("Image decoding failed at slot %s: %s", idx, exc)
                    meta["detail"] = "Unsupported or corrupted image."
                    meta["error_code"] = 400
                except Exception as exc:
                    logger.exception("Unhandled error while decoding upload at slot %s", idx)
                    meta["detail"] = f"Failed to read image: {exc}"
                    meta["error_code"] = 400

            if meta["detail"] is None and pil_img is not None:
                backend = "opencv"
                fallback_reason: Optional[str] = None
                enhanced: Optional[Image.Image] = None

                if DEFAULT_USE_REALESRGAN:
                    loop = asyncio.get_running_loop()
                    try:
                        enhanced = await loop.run_in_executor(
                            None,
                            enhance_with_realesrgan,
                            pil_img,
                            DEFAULT_SCALE,
                        )
                        backend = "realesrgan"
                    except Exception as exc:
                        fallback_reason = str(exc).replace("\n", " ")
                        logger.warning(
                            "Real-ESRGAN inference failed at slot %s; falling back to OpenCV pipeline: %s",
                            idx,
                            exc,
                        )

                if enhanced is None and meta["detail"] is None:
                    try:
                        enhanced = enhance_image(
                            pil_img,
                            scale=DEFAULT_SCALE,
                            denoise_strength=DEFAULT_DENOISE,
                            sharpen_amount=DEFAULT_SHARPEN,
                            contrast_clip_limit=DEFAULT_CONTRAST,
                        )
                    except ValueError as exc:
                        logger.warning("Semantic enhancement failure at slot %s: %s", idx, exc)
                        meta["detail"] = f"Could not process image: {exc}"
                        meta["error_code"] = 422
                    except Exception as exc:
                        logger.exception("OpenCV enhancement pipeline crashed at slot %s", idx)
                        meta["detail"] = f"Enhancement failed: {exc}"
                        meta["error_code"] = 500

                if meta["detail"] is None and enhanced is not None:
                    buffer = BytesIO()
                    try:
                        enhanced.save(buffer, format="PNG", optimize=True)
                    except Exception as exc:
                        logger.exception("Encoding enhanced image failed at slot %s", idx)
                        meta["detail"] = f"Failed to encode image: {exc}"
                        meta["error_code"] = 500
                    else:
                        payload = buffer.getvalue()
                        stem = Path(meta["filename"]).stem or f"image_{idx + 1}"
                        meta.update(
                            status="success",
                            detail=None,
                            error_code=None,
                            backend=backend,
                            warning=fallback_reason[:120] if fallback_reason else None,
                            content_type="image/png",
                            size=len(payload),
                            output_filename=f"{stem}_enhanced.png",
                        )
        finally:
            try:
                await upload.close()
            except Exception:
                pass

        if payload is not None:
            meta["_data"] = payload

        results.append(meta)

    success_entries = [entry for entry in results if entry["status"] == "success"]
    warning_entries = [entry for entry in success_entries if entry.get("warning")]
    error_entries = [entry for entry in results if entry["status"] != "success"]

    if len(results) == 1:
        entry = results[0]
        if entry["status"] == "success":
            data = entry["_data"]
            headers = {"X-Enhancer-Backend": entry["backend"]}
            status_code = 207 if entry.get("warning") else 200
            if entry.get("warning"):
                headers["X-Enhancer-Warning"] = entry["warning"]
            logger.info(
                "Enhancement completed with backend=%s status=%s",
                entry["backend"],
                status_code,
            )
            return StreamingResponse(
                BytesIO(data),
                media_type="image/png",
                headers=headers,
                status_code=status_code,
            )

        status_code = int(entry.get("error_code") or 400)
        detail = entry.get("detail") or "Could not process image."
        logger.info("Enhancement request failed status=%s reason=%s", status_code, detail)
        raise HTTPException(status_code=status_code, detail=detail)

    if success_entries:
        status_code = 207 if error_entries or warning_entries else 200
    else:
        status_code = max((int(entry.get("error_code") or 400) for entry in results), default=400)

    headers: dict[str, str] = {}
    if success_entries:
        headers["X-Enhancer-Backends"] = ",".join(
            sorted({entry["backend"] for entry in success_entries if entry.get("backend")})
        )
    if warning_entries:
        warnings = [entry["warning"] for entry in warning_entries if entry.get("warning")]
        if warnings:
            headers["X-Enhancer-Warnings"] = " | ".join(warnings)[:240]

    response_items = []
    for entry in results:
        serializable = {k: v for k, v in entry.items() if not k.startswith("_") and v is not None}
        if entry["status"] == "success":
            serializable["data"] = base64.b64encode(entry["_data"]).decode("ascii")
        response_items.append(serializable)

    for entry in results:
        entry.pop("_data", None)

    logger.info(
        "Enhancement batch completed total=%s successes=%s failures=%s status=%s",
        len(results),
        len(success_entries),
        len(error_entries),
        status_code,
    )

    body = {
        "results": response_items,
        "success": len(success_entries),
        "failed": len(error_entries),
    }

    return JSONResponse(status_code=status_code, content=body, headers=headers)
