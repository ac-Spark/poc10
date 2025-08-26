# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import traceback
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Load environment variables first
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"Environment loaded. HF_TOKEN: {'Yes' if os.getenv('HF_TOKEN') else 'No'}")
except ImportError:
    print("dotenv not available, skipping .env file loading")

# Import from root-level modules
from api_models import GenerationRequest, GenerationResponse, StatusResponse, HealthResponse
from logger_utils import build_logger
from constants import (
    SERVER_ERROR_MSG, DEFAULT_SAVE_DIR, API_TITLE, API_DESCRIPTION, 
    API_VERSION, API_CONTACT, API_LICENSE_INFO, API_TAGS_METADATA
)
from model_worker import ModelWorker

# Global variables
SAVE_DIR = DEFAULT_SAVE_DIR
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

def clean_old_tasks(max_tasks=30):
    """
    Maintain a maximum number of task folders in SAVE_DIR.
    
    If the number of existing folders exceeds max_tasks, the oldest folders are removed.
    
    Args:
        max_tasks (int): Maximum number of task folders to keep. Defaults to 30.
    """
    try:
        import shutil
        from pathlib import Path
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]
        
        if len(dirs) > max_tasks:
            # Sort by creation time, oldest first
            dirs.sort(key=lambda x: x.stat().st_ctime)
            
            # Remove oldest folders until we're within the limit
            folders_to_remove = len(dirs) - max_tasks
            for i in range(folders_to_remove):
                try:
                    shutil.rmtree(dirs[i])
                    logger.info(f"Removed old task folder: {dirs[i]}")
                except Exception as e:
                    logger.warning(f"Failed to remove old task folder {dirs[i]}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during task folder cleanup: {e}")

# Global worker and semaphore instances
worker = None
model_semaphore = None


def process_image_input(image_base64: Optional[str] = None, image_file: Optional[UploadFile] = None) -> str:
    """Process image input from either base64 string or uploaded file."""
    if image_base64:
        return image_base64
    elif image_file:
        try:
            # Read the uploaded file
            image_data = image_file.file.read()
            # Convert to base64
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing uploaded image: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either image base64 string or image file must be provided")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    tags_metadata=API_TAGS_METADATA
)

app.mount("/static", StaticFiles(directory="assets"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", tags=["generation"])
async def generate_3d_model(
    image: Optional[str] = Form(None, description="Base64 encoded image"),
    image_file: Optional[UploadFile] = File(None, description="Image file upload"),
    remove_background: bool = Form(True, description="Remove background from image"),
    texture: bool = Form(False, description="Generate textures for 3D model"),
    seed: int = Form(1234, description="Random seed", ge=0, le=2**32-1),
    octree_resolution: int = Form(256, description="Octree resolution", ge=64, le=512),
    num_inference_steps: int = Form(5, description="Inference steps", ge=1, le=20),
    guidance_scale: float = Form(5.0, description="Guidance scale", ge=0.1, le=20.0),
    num_chunks: int = Form(8000, description="Number of chunks", ge=1000, le=20000),
    face_count: int = Form(40000, description="Face count", ge=1000, le=100000)
):
    """
    Generate a 3D model from an input image.
    
    This endpoint takes an image and generates a 3D model with optional textures.
    The generation process includes background removal, mesh generation, and optional texture mapping.
    
    Accepts either:
    - Base64 encoded image string via 'image' parameter
    - Direct file upload via 'image_file' parameter
    
    Returns:
        FileResponse: The generated 3D model file (GLB or OBJ format)
    """
    logger.info("Worker generating...")
    
    try:
        # Process image input
        image_base64 = process_image_input(image, image_file)
        
        # Build parameters dict
        params = {
            "image": image_base64,
            "remove_background": remove_background,
            "texture": texture,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_chunks": num_chunks,
            "face_count": face_count
        }
        
        uid = uuid.uuid4()
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except HTTPException:
        raise  # Re-raise HTTP exceptions from process_image_input
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Caught ValueError: {e}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        logger.error(f"Caught torch.cuda.CudaError: {e}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        logger.error(f"Caught Unknown Error: {e}")
        traceback.print_exc()
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send", response_model=GenerationResponse, tags=["generation"])
async def send_generation_task(request: GenerationRequest):
    """
    Send a 3D generation task to be processed asynchronously (JSON format).
    
    This endpoint starts the generation process in the background and returns a task ID.
    Use the /status/{uid} endpoint to check the progress and retrieve the result.
    
    Returns:
        GenerationResponse: Contains the unique task identifier
    """
    logger.info("Worker send...")
    
    # Convert Pydantic model to dict for compatibility
    params = request.dict()
    
    if not params.get("image"):
        raise HTTPException(status_code=400, detail="Image base64 string is required")
    
    # Clean old tasks before starting new one
    clean_old_tasks(max_tasks=30)
    
    uid = uuid.uuid4()
    try:
        threading.Thread(target=worker.generate, args=(uid, params,)).start()
        ret = {"uid": str(uid)}
        return JSONResponse(ret, status_code=200)
    except Exception as e:
        logger.error(f"Failed to start generation thread: {e}")
        ret = {"error": "Failed to start generation"}
        return JSONResponse(ret, status_code=500)


@app.post("/send_file", response_model=GenerationResponse, tags=["generation"])
async def send_generation_task_file(
    image_front: Optional[UploadFile] = File(None, description="Front view image file upload"),
    image_back: Optional[UploadFile] = File(None, description="Back view image file upload"),
    image_left: Optional[UploadFile] = File(None, description="Left view image file upload"),
    image_right: Optional[UploadFile] = File(None, description="Right view image file upload"),
    remove_background: bool = Form(True, description="Remove background from image"),
    texture: bool = Form(False, description="Generate textures for 3D model"),
    seed: int = Form(1234, description="Random seed", ge=0, le=2**32-1),
    octree_resolution: int = Form(256, description="Octree resolution", ge=64, le=512),
    num_inference_steps: int = Form(5, description="Inference steps", ge=1, le=20),
    guidance_scale: float = Form(5.0, description="Guidance scale", ge=0.1, le=20.0),
    num_chunks: int = Form(8000, description="Number of chunks", ge=1000, le=20000),
    face_count: int = Form(40000, description="Face count", ge=1000, le=100000),
    output_format: str = Form("glb", description="Output format (glb or obj)", regex="^(glb|obj)$")
):
    """
    Send a 3D generation task with one or more view images to be processed asynchronously.
    
    This endpoint starts the generation process in the background and returns a task ID.
    Use the /status/{uid} endpoint to check the progress and retrieve the result.
    
    Returns:
        GenerationResponse: Contains the unique task identifier
    """
    logger.info("Worker send_file (multi-view)...")
    logger.info(f"Received files - front: {image_front is not None}, back: {image_back is not None}, left: {image_left is not None}, right: {image_right is not None}")
    logger.info(f"Parameters - texture: {texture}, seed: {seed}, steps: {num_inference_steps}")
    
    try:
        images = {
            "front": image_front,
            "back": image_back,
            "left": image_left,
            "right": image_right,
        }

        image_params = {}
        original_filenames = {}
        for view, upload_file in images.items():
            if upload_file and upload_file.filename:
                image_data = await upload_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                image_params[view] = image_base64
                original_filenames[view] = upload_file.filename

        if not image_params:
            logger.error("No valid image files received")
            raise HTTPException(status_code=400, detail="At least one image must be provided")

        # Build parameters dict
        import datetime
        params = {
            "image": image_params,
            "remove_background": remove_background,
            "texture": texture,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_chunks": num_chunks,
            "face_count": face_count,
            "output_format": output_format,
            "original_filenames": original_filenames,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Clean old tasks before starting new one
        clean_old_tasks(max_tasks=30)
        
        uid = uuid.uuid4()
        threading.Thread(target=worker.generate, args=(uid, params,)).start()
        ret = {"uid": str(uid)}
        return JSONResponse(ret, status_code=200)
    except HTTPException as e:
        logger.error(f"HTTP Exception in send_file: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in send_file: {e}")
        traceback.print_exc()
        return JSONResponse({"error": f"Failed to start generation: {str(e)}"}, status_code=500)


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        HealthResponse: Service health status and worker identifier
    """
    return JSONResponse({"status": "healthy", "worker_id": worker_id}, status_code=200)


@app.get("/status/{uid}", response_model=StatusResponse, tags=["status"])
async def status(uid: str):
    """
    Check the status of a generation task.
    
    Args:
        uid: The unique identifier of the generation task
        
    Returns:
        StatusResponse: Current status of the task and result if completed
    """
    task_folder = os.path.join(SAVE_DIR, str(uid))
    status_file = os.path.join(task_folder, 'status.json')

    if not os.path.exists(task_folder) or not os.path.exists(status_file):
        # Fallback for tasks that started before this change or if status file is missing
        # Check for final files as a last resort (try both glb and obj)
        for ext in ['glb', 'obj']:
            textured_file_path = os.path.join(task_folder, f'textured_mesh.{ext}')
            white_file_path = os.path.join(task_folder, f'white_mesh.{ext}')
            if os.path.exists(textured_file_path) or os.path.exists(white_file_path):
                file_path = textured_file_path if os.path.exists(textured_file_path) else white_file_path
                try:
                    base64_str = base64.b64encode(open(file_path, 'rb').read()).decode()
                    return JSONResponse({'status': 'completed', 'progress': 100, 'model_base64': base64_str})
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    return JSONResponse({'status': 'error', 'progress': 0, 'message': 'Failed to read generated file'})
        
        return JSONResponse({'status': 'processing', 'progress': 10, 'message': 'Task folder exists, but status is unknown.'})

    try:
        with open(status_file, 'r') as f:
            import json
            status_data = json.load(f)
        
        if status_data.get('stage') == 'completed':
            # Get output format from status data, default to 'glb'
            output_format = status_data.get('output_format', 'glb')
            
            textured_file_path = os.path.join(task_folder, f'textured_mesh.{output_format}')
            white_file_path = os.path.join(task_folder, f'white_mesh.{output_format}')
            
            file_path = None
            if os.path.exists(textured_file_path):
                file_path = textured_file_path
            elif os.path.exists(white_file_path):
                file_path = white_file_path

            if file_path:
                base64_str = base64.b64encode(open(file_path, 'rb').read()).decode()
                status_data['model_base64'] = base64_str
            else:
                status_data = {'status': 'error', 'progress': 0, 'message': 'Completed, but model file not found.'}

        return JSONResponse(status_data)

    except Exception as e:
        logger.error(f"Error reading status file for {uid}: {e}")
        return JSONResponse({'status': 'error', 'progress': 0, 'message': 'Failed to read status file.'}, status_code=500)


@app.get("/download/{uid}", tags=["status"])
async def download_model(uid: str):
    """
    Download the generated 3D model file directly.
    
    Args:
        uid: The unique identifier of the generation task
        
    Returns:
        FileResponse: The generated 3D model file for download
    """
    # Check for new folder structure first
    task_folder = os.path.join(SAVE_DIR, str(uid))
    textured_file_path = os.path.join(task_folder, 'textured_mesh.glb')
    white_file_path = os.path.join(task_folder, 'white_mesh.glb')
    
    # Also check old structure for backward compatibility
    old_textured_path = os.path.join(SAVE_DIR, f'{uid}_textured.glb')
    old_initial_path = os.path.join(SAVE_DIR, f'{uid}_initial.glb')
    
    # If textured file exists (new structure), return it
    if os.path.exists(textured_file_path):
        return FileResponse(
            textured_file_path, 
            filename=f"{uid}_textured.glb",
            media_type="application/octet-stream"
        )
    
    # If white mesh exists (new structure), return it
    elif os.path.exists(white_file_path):
        return FileResponse(
            white_file_path, 
            filename=f"{uid}_white.glb",
            media_type="application/octet-stream"
        )
    
    # Backward compatibility: check old structure
    elif os.path.exists(old_textured_path):
        return FileResponse(
            old_textured_path, 
            filename=f"{uid}_textured.glb",
            media_type="application/octet-stream"
        )
    
    elif os.path.exists(old_initial_path):
        return FileResponse(
            old_initial_path, 
            filename=f"{uid}_initial.glb",
            media_type="application/octet-stream"
        )
    
    # If no files exist, return error
    else:
        raise HTTPException(status_code=404, detail="Model file not found or still processing")


@app.get("/models", tags=["models"])
async def get_models():
    """
    Get a list of all generated models.

    Returns:
        JSONResponse: A list of model identifiers.
    """
    try:
        models = [d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d))]
        return JSONResponse({"models": models})
    except Exception as e:
        logger.error(f"Error listing models in {SAVE_DIR}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")


@app.get("/models/{uid}/view", tags=["models"])
async def view_model(uid: str):
    """
    Get the HTML view for a specific model.

    Args:
        uid: The unique identifier of the generation task

    Returns:
        FileResponse: The HTML file for viewing the model.
    """
    task_folder = os.path.join(SAVE_DIR, str(uid))
    html_path = os.path.join(task_folder, 'white_mesh.html')

    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Model view not found")


@app.get("/models/{uid}/white_mesh.glb", tags=["models"])
async def get_model_file(uid: str):
    """
    Get the GLB model file for a specific model.

    Args:
        uid: The unique identifier of the generation task

    Returns:
        FileResponse: The GLB model file.
    """
    task_folder = os.path.join(SAVE_DIR, str(uid))
    glb_path = os.path.join(task_folder, 'white_mesh.glb')

    if os.path.exists(glb_path):
        return FileResponse(glb_path, media_type="application/octet-stream")
    else:
        raise HTTPException(status_code=404, detail="Model file not found")


@app.get("/static/env_maps/gradient.jpg", tags=["static"])
async def get_gradient_env():
    """
    Serve the gradient environment map for model viewer.

    Returns:
        Response: A simple gradient response or default environment.
    """
    # Return a simple solid color response since we don't have the actual gradient file
    return JSONResponse({"message": "Environment map not available"}, status_code=404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--cache-path', type=str, default='./save_dir')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Update SAVE_DIR based on cache-path argument
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(
        model_path=args.model_path, 
        subfolder=args.subfolder,
        device=args.device, 
        low_vram_mode=args.low_vram_mode,
        worker_id=worker_id,
        model_semaphore=model_semaphore,
        save_dir=SAVE_DIR
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
