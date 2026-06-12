#!/usr/bin/env python3
import argparse

import uvicorn
from fastapi import FastAPI, HTTPException

from gui_api import detect_charuco_image, run_calibration


app = FastAPI(title="Hand-eye calibration local API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect-charuco")
def detect_charuco(payload: dict):
    try:
        return detect_charuco_image(
            payload["imagePath"],
            camera_params=payload.get("cameraParams"),
            output_dir=payload.get("outputDir"),
            squares_x=payload.get("squaresX", 11),
            squares_y=payload.get("squaresY", 8),
            square_length=payload.get("squareLength", 0.014),
            marker_length=payload.get("markerLength", 0.010),
            aruco_dict=payload.get("arucoDict", "DICT_5X5_100"),
            depth_path=payload.get("depthPath"),
            camera_intrinsics=payload.get("cameraIntrinsics"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing required field: {exc.args[0]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run-calibration")
def run_calibration_endpoint(payload: dict):
    try:
        return run_calibration(payload)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing required field: {exc.args[0]}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description="Hand-eye calibration local API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
