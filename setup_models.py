"""
Model Setup Utility for Smart Traffic Management System.

Downloads and validates pretrained YOLOv8 Nano ONNX weights (12.8 MB)
from the official Ultralytics release assets repository.

Usage:
    python setup_models.py
    python setup_models.py --force
    python setup_models.py --check

Licensing & Attribution:
    - Architecture: Ultralytics YOLOv8 (AGPL-3.0 / Enterprise)
    - Dataset: Microsoft COCO (80 classes, CC BY 4.0)
    - Target Classes: Car (2), Motorcycle (3), Bus (5), Truck (7)
"""

import sys
import argparse
import urllib.request
import urllib.error
import time
from pathlib import Path

# Primary official Ultralytics release asset URL
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
FALLBACK_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"

TARGET_DIR = Path(__file__).resolve().parent / "models"
TARGET_FILE = TARGET_DIR / "yolov8n.onnx"
EXPECTED_MIN_BYTES = 10 * 1024 * 1024  # At least 10 MB (approx 12.8 MB)


def check_model() -> bool:
    """Verifies if the model exists and satisfies minimum size constraints."""
    if not TARGET_FILE.exists():
        print(f"[-] Model weights not found at: {TARGET_FILE}")
        return False
    size_bytes = TARGET_FILE.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes < EXPECTED_MIN_BYTES:
        print(f"[!] Warning: Model file exists but is unexpectedly small ({size_mb:.2f} MB < 10 MB).")
        return False
    print(f"[+] Model weights verified at: {TARGET_FILE} ({size_mb:.2f} MB)")
    return True


def download_model(force: bool = False, timeout: int = 60) -> bool:
    """
    Downloads the YOLOv8 Nano ONNX model with progress indicator and error handling.
    
    Args:
        force: Overwrite existing valid weights if True.
        timeout: Socket timeout in seconds.
        
    Returns:
        True if download and verification succeeded, False otherwise.
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if not force and check_model():
        print("[*] Model weights already exist and are valid. Use --force to re-download.")
        return True

    print("\n" + "=" * 70)
    print("🚦 Smart Traffic Management System — Model Setup")
    print("=" * 70)
    print(f"Destination: {TARGET_FILE}")
    print("Network Requirement: Outbound HTTPS access to github.com")
    print("Target Model: YOLOv8 Nano ONNX (~12.8 MB, 640x640 input)")
    print("-" * 70)

    urls_to_try = [MODEL_URL, FALLBACK_URL]
    temp_target = TARGET_FILE.with_suffix(".tmp")

    for idx, url in enumerate(urls_to_try, 1):
        print(f"[*] Attempting download from source {idx}/{len(urls_to_try)}:\n    {url}")
        try:
            start_time = time.time()

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SmartTrafficManagement-ModelSetup/1.0"}
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 256  # 256 KB chunks

                with open(temp_target, "wb") as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r    Progress: {pct:5.1f}% ({mb_down:5.1f}/{mb_total:5.1f} MB)")
                            sys.stdout.flush()

            sys.stdout.write("\n")
            elapsed = time.time() - start_time
            print(f"[+] Download finished in {elapsed:.2f}s.")

            # Validate temporary file size
            temp_size = temp_target.stat().st_size
            if temp_size < EXPECTED_MIN_BYTES:
                print(f"[-] Downloaded file is too small ({temp_size} bytes). Discarding.")
                if temp_target.exists():
                    temp_target.unlink()
                continue

            # Atomic rename
            if TARGET_FILE.exists():
                TARGET_FILE.unlink()
            temp_target.rename(TARGET_FILE)

            print(f"[✔] Success! Pretrained model saved to:\n    {TARGET_FILE.resolve()}")
            print("[✔] You can now start the application with: streamlit run app.py\n")
            return True

        except urllib.error.HTTPError as e:
            print(f"[-] HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            print(f"[-] Network connection error: {e.reason}")
            print("    Please check your internet connection or proxy settings.")
        except TimeoutError:
            print(f"[-] Connection timed out after {timeout} seconds.")
        except Exception as e:
            print(f"[-] Unexpected error during download: {type(e).__name__}: {e}")
        finally:
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except Exception:
                    pass

    print("\n" + "!" * 70)
    print("⚠️  MODEL DOWNLOAD FAILED")
    print("!" * 70)
    print("The system could not download YOLOv8 Nano weights automatically.")
    print("Offline / Fallback Options:")
    print("  1. The application will automatically fall back to:")
    print("     - Classical Computer Vision (OpenCV MOG2 background subtraction)")
    print("     - Procedural Synthetic Simulation (zero external files required)")
    print("  2. Manual Setup:")
    print(f"     Download 'yolov8n.onnx' from: {MODEL_URL}")
    print(f"     Place the file into: {TARGET_DIR.resolve()}")
    print("=" * 70 + "\n")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup pretrained model weights for Smart Traffic Management.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model already exists.")
    parser.add_argument("--check", action="store_true", help="Check model status without downloading.")
    parser.add_argument("--timeout", type=int, default=60, help="Download timeout in seconds (default: 60).")
    args = parser.parse_args()

    if args.check:
        success = check_model()
        sys.exit(0 if success else 1)
    else:
        success = download_model(force=args.force, timeout=args.timeout)
        sys.exit(0 if success else 1)
