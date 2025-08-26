"""
Model worker for Hunyuan3D API server.
"""
import os
import time
import uuid
import base64
import trimesh
from io import BytesIO
from PIL import Image
import torch

# Apply torchvision compatibility fix before other imports
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    """Generate HTML viewer for GLB model (adapted from gradio_app.py)"""
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    
    # Simple HTML template if asset files don't exist
    template_html = f"""<!DOCTYPE html>
<html>
<head>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
        model-viewer {{
            width: {width}px;
            height: {height}px;
            background-color: #f0f0f0;
        }}
    </style>
</head>
<body>
    <model-viewer src="{related_path}" alt="3D Model" auto-rotate camera-controls></model-viewer>
</body>
</html>"""
    
    # Try to use template file if it exists
    try:
        if os.path.exists(template_name):
            with open(template_name, 'r', encoding='utf-8') as f:
                template_html = f.read()
                template_html = template_html.replace('#height#', str(height))
                template_html = template_html.replace('#width#', str(width))
                template_html = template_html.replace('#src#', related_path)
    except Exception as e:
        logger.warning(f"Could not load template {template_name}: {e}")
    
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(template_html)
    
    logger.info(f"Generated HTML viewer: {output_html_path}")
    return output_html_path


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.
    
    Args:
        image (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    """
    Worker class for handling 3D model generation tasks.
    """
    
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1',
                 device='cuda',
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache'):
        """
        Initialize the model worker.
        
        Args:
            model_path (str): Path to the shape generation model
            subfolder (str): Subfolder containing the model files
            device (str): Device to run the model on ('cuda' or 'cpu')
            low_vram_mode (bool): Whether to use low VRAM mode
            worker_id (str): Unique identifier for this worker
            model_semaphore: Semaphore for controlling model concurrency
            save_dir (str): Directory to save generated files
        """
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        
        logger.info(f"Loading the model {model_path} on worker {self.worker_id} ...")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize shape generation pipeline (matching demo.py)
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        
        # Initialize texture generation pipeline (matching demo.py)
        # Initialize paint pipeline for texture generation
        # Initialize paint pipeline only when needed (lazy loading)
        # This saves GPU memory when texture generation is not requested
        self.paint_pipeline = None
        self._paint_pipeline_config = None
        
        # Prepare config for lazy loading
        try:
            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            max_num_view = 8
            resolution = 768
            conf = Hunyuan3DPaintConfig(max_num_view, resolution)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            self._paint_pipeline_config = (Hunyuan3DPaintPipeline, conf)
            logger.info("Paint pipeline config prepared for lazy loading with full resolution.")
        except Exception as e:
            logger.error(f"Failed to prepare paint pipeline config: {e}")
            self._paint_pipeline_config = None
        # Ensure save_dir exists
        os.makedirs(self.save_dir, exist_ok=True)
            
    def get_queue_length(self):
        """
        Get the current queue length for model processing.
        
        Returns:
            int: Number of tasks in the queue
        """
        if self.model_semaphore is None:
            return 0
        else:
            return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
                   (len(self.model_semaphore._waiters) if hasattr(self.model_semaphore, '_waiters') and self.model_semaphore._waiters is not None else 0)

    def get_status(self):
        """
        Get the current status of the worker.
        
        Returns:
            dict: Status information including speed and queue length
        """
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Generate a 3D model from the given parameters.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        start_time = time.time()
        logger.info(f"Generating 3D model for uid: {uid}")

        task_folder = os.path.join(self.save_dir, str(uid))
        os.makedirs(task_folder, exist_ok=True)
        status_file = os.path.join(task_folder, 'status.json')

        def write_status(stage, progress, message=""):
            with open(status_file, 'w') as f:
                import json
                status_data = {
                    'stage': stage, 
                    'progress': progress, 
                    'message': message,
                    'timestamp': params.get('timestamp'),
                    'original_filenames': params.get('original_filenames')
                }
                # Save output format info for API endpoints to use
                if 'output_format' in params:
                    status_data['output_format'] = params['output_format']
                json.dump(status_data, f)

        try:
            write_status('starting', 5, "Starting generation task...")

            # Handle input image(s)
            image_input = params.get("image")
            if not image_input:
                raise ValueError("No input image provided")

            if isinstance(image_input, dict):
                # Multi-view case
                image = {
                    view: load_image_from_base64(base64_str)
                    for view, base64_str in image_input.items()
                }
            elif isinstance(image_input, str):
                # Single-view case
                image = load_image_from_base64(image_input)
            else:
                raise TypeError(f"Unsupported image format: {type(image_input)}")

            # Convert to RGBA and remove background if needed
            if params.get('remove_background', True):
                if isinstance(image, dict):
                    # Multi-view case
                    for view, img in image.items():
                        image[view] = self.rembg(img.convert("RGBA"))
                else:
                    # Single-view case
                    image = self.rembg(image.convert("RGBA"))

            write_status('generating_shape', 20, "Generating 3D shape...")

            # Define progress callback
            num_inference_steps = params.get('num_inference_steps', 5)
            texture_enabled = params.get('texture', False)
            
            def progress_callback(step, timestep, latents):
                # Shape generation: 5% -> 80% (if texture) or 95% (if no texture)
                progress_range = 75 if texture_enabled else 90
                current_progress = int(5 + (step / num_inference_steps) * progress_range)
                write_status('generating_shape', current_progress, f"Generating shape: step {step}/{num_inference_steps}")

            # Generate mesh 
            mesh = self.pipeline(
                image=image if not isinstance(image, dict) else image.get('front'),
                num_inference_steps=num_inference_steps,
                callback_steps=1,
                callback=progress_callback
            )[0]
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

            # Remove background plane by keeping only the largest connected component
            mesh_components = mesh.split(only_watertight=False)
            if mesh_components:
                mesh = sorted(mesh_components, key=lambda x: x.area, reverse=True)[0]

            # Export initial mesh without texture (following gradio naming)
            output_format = params.get('output_format', 'glb')
            initial_save_path = os.path.join(task_folder, f'white_mesh.{output_format}')
            mesh.export(initial_save_path)
            
            # Generate HTML viewer
            build_model_viewer_html(task_folder, textured=False)
            
            # Try to generate textured mesh (only if requested)
            textured_save_path = None
            if texture_enabled:
                write_status('generating_texture', 80, "Generating texture...")
                # Lazy load paint pipeline when needed
                if self.paint_pipeline is None and self._paint_pipeline_config is not None:
                    try:
                        # Clear shape generation from GPU memory first
                        if hasattr(self.pipeline, 'to'):
                            self.pipeline.to('cpu')
                        torch.cuda.empty_cache()
                        
                        # Load paint pipeline
                        logger.info("Lazy loading paint pipeline...")
                        PipelineClass, conf = self._paint_pipeline_config
                        self.paint_pipeline = PipelineClass(conf)
                        logger.info("Paint pipeline loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to lazy load paint pipeline: {e}")
                        self.paint_pipeline = None
                
                if self.paint_pipeline is not None:
                    logger.info("Starting texture generation...")
                    # Generate textured mesh as obj (as in demo)
                    output_mesh_path_obj = os.path.join(task_folder, 'textured_mesh.obj')
                    
                    # TODO: Add progress callback for texture generation if available
                    # For now, we'll just set progress to a fixed value during this stage
                    write_status('generating_texture', 85, "Applying texture to the model...")

                    textured_path_obj = self.paint_pipeline(
                        mesh_path=initial_save_path,
                        image_path=image if not isinstance(image, dict) else image.get('front'), # Paint pipeline might need a single image
                        output_mesh_path=output_mesh_path_obj,
                        save_glb=False            
                    )
                    
                    # Convert to output format
                    if output_format == 'glb':
                        textured_save_path = os.path.join(task_folder, 'textured_mesh.glb')
                        quick_convert_with_obj2gltf(textured_path_obj, textured_save_path)
                    else:
                        # For OBJ format, just copy the original OBJ file
                        textured_save_path = os.path.join(task_folder, 'textured_mesh.obj')
                        import shutil
                        shutil.copy2(textured_path_obj, textured_save_path)
                    
                    # Generate HTML viewer for textured version
                    build_model_viewer_html(task_folder, textured=True)
                    
                    logger.info("---Texture generation takes %s seconds ---" % (time.time() - start_time))
                    logger.info(f"Generated textured mesh: {textured_save_path}")
                    
                    # Move paint pipeline back to CPU to free GPU memory
                    if hasattr(self.paint_pipeline, 'to'):
                        self.paint_pipeline.to('cpu')
                    # Move shape pipeline back to GPU
                    if hasattr(self.pipeline, 'to'):
                        self.pipeline.to('cuda')
                    torch.cuda.empty_cache()

            write_status('completed', 100, "Generation complete!")

        except Exception as e:
            logger.error(f"Generation failed for uid {uid}: {e}")
            write_status('error', 0, message=str(e))
            # Re-raise the exception to be handled by the calling thread in api_server
            raise

        if self.low_vram_mode:
            torch.cuda.empty_cache()
            
        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        
        # Return the best available mesh (textured if available, otherwise white)
        final_path = textured_save_path if textured_save_path and os.path.exists(textured_save_path) else initial_save_path
        return final_path, uid 