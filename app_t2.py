import os
import sys
import math
import argparse
import random
import torch
import numpy as np
import gradio as gr
import imageio
import subprocess
import shutil
import tempfile
import uuid
import time
from PIL import Image
from einops import rearrange, repeat
import torch.nn.functional as F
import torchvision.transforms as TT

# Import existing modules
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
import diffusion_video
from arguments import get_args
import decord
from decord import VideoReader
from data_video import resize_for_rectangle_crop

# -----------------------------------------------------------------------------
# 0. LOGGING INFRASTRUCTURE (Uvicorn/Gradio)
# -----------------------------------------------------------------------------
LOG_FILE = "app_monitor.log"

class DualLogger(object):
    """Writes to both the actual terminal and a log file simultaneously."""
    def __init__(self, original_stream):
        self.terminal = original_stream
        self.log = open(LOG_FILE, "a")

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
            self.log.flush()
        except: pass # Ignore log file errors to keep app running

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    # Pass these calls to the real terminal so Uvicorn doesn't crash
    def isatty(self):
        return self.terminal.isatty()
        
    def fileno(self):
        return self.terminal.fileno()

# Redirect stdout/stderr
if not isinstance(sys.stdout, DualLogger):
    # Reset log file on startup
    with open(LOG_FILE, "w") as f: f.write("--- App Started ---\n")
    sys.stdout = DualLogger(sys.stdout)
    sys.stderr = DualLogger(sys.stderr)

def read_logs():
    """Reads the last 4000 characters from the log file for the UI."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                f.seek(0, 2) 
                fsize = f.tell()
                f.seek(max(fsize - 4000, 0), 0)
                return f.read()
        except: return "Reading logs..."
    return "Waiting for logs..."

# -----------------------------------------------------------------------------
# 1. HELPERS
# -----------------------------------------------------------------------------

def load_image_to_tensor_chw_normalized(pil_image):
    transform = TT.Compose([TT.ToTensor()])
    image_tensor = transform(pil_image)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)
    return image_tensor

def load_video_to_tensor(video_path):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=-1, width=-1)
    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    tensor = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    return tensor

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T=None, device="cuda"):
    batch, batch_uc = {}, {}
    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        elif key in ["original_size_as_tuple", "target_size_as_tuple", "crop_coords_top_left"]:
            val = [value_dict["orig_height"], value_dict["orig_width"]] if "size" in key else [0, 0]
            batch[key] = torch.tensor(val).to(device).repeat(*N, 1)
        elif key == "aesthetic_score":
            batch[key] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            batch_uc[key] = torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
        elif key in ["fps", "fps_id", "motion_bucket_id", "cond_aug"]:
            batch[key] = torch.tensor([value_dict.get(key, 16)]).to(device).repeat(math.prod(N))
        else:
            batch[key] = value_dict[key]
    if T is not None: batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

# -----------------------------------------------------------------------------
# 2. MODEL INIT
# -----------------------------------------------------------------------------
print("Initializing Generation Model...")
sys.argv = ["app.py", "--base", "configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml", "configs/sampling/wan_pose_14Bsc_xc_cli.yaml"]
py_parser = argparse.ArgumentParser(add_help=False)
known, args_list = py_parser.parse_known_args()
args = get_args(args_list)
args = argparse.Namespace(**vars(args), **vars(known))
args.model_config.network_config.params.transformer_args.checkpoint_activations = False
model = get_model(args, diffusion_video.SATVideoDiffusionEngine)
load_checkpoint(model, args)
model.eval()
print("✅ Model Loaded!")

# -----------------------------------------------------------------------------
# 3. PIPELINE FUNCTIONS
# -----------------------------------------------------------------------------

# --- A. Original Unified Pipeline (Video Gen) ---
def run_scail_pipeline(
    prompt, driving_video, ref_image, 
    seed, steps, cfg, fps, slider_w, slider_h, 
    use_align, is_multi
):
    # Phase 1: Setup
    if driving_video is None or ref_image is None:
        raise gr.Error("Missing Inputs! Please upload both Driving Video and Reference Image.")

    print(f"\n--- NEW JOB STARTED: {prompt} ---")
    
    staging_root = os.path.abspath("gradio_outputs")
    os.makedirs(staging_root, exist_ok=True)
    job_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(staging_root, job_id)
    os.makedirs(work_dir, exist_ok=True)

    drive_path = os.path.join(work_dir, "driving.mp4")
    ref_path = os.path.join(work_dir, "ref.jpg")
    shutil.copy2(driving_video, drive_path)
    shutil.copy2(ref_image, ref_path)

    # Phase 2: Extract Pose
    print("STEP 1/3: Extracting Pose...")
    script_name = "process_pose_multi.py" if is_multi else "process_pose.py"
    align_flag = "--use_align" if use_align else ""
    
    command = (
        f"cd SCAIL-Pose && "
        f"eval \"$(conda shell.bash hook)\" && "
        f"conda activate ./SCAIL_poses && "
        f"python NLFPoseExtract/{script_name} --subdir \"{work_dir}\" --resolution {slider_h} {slider_w} {align_flag}"
    )
    
    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    except subprocess.CalledProcessError:
        raise gr.Error("Pose Extraction Failed! Check logs.")

    pose_path = None
    for fname in ["rendered_aligned.mp4", "rendered.mp4", "pose.mp4"]:
        if os.path.exists(os.path.join(work_dir, fname)):
            pose_path = os.path.join(work_dir, fname)
            break
    
    if not pose_path: raise gr.Error("Pose Extraction finished but no output file found.")

    print("STEP 2/3: Loading Tensors...")

    # Phase 3: Inference
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    if seed == -1: seed = random.randint(0, 10**6)
    torch.manual_seed(seed)
    
    try:
        model.sampler_config.params.num_steps = int(steps)
        model.sampler_config.params.guider_config.params.scale = float(cfg)
    except: pass

    pil_img = Image.open(ref_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    
    dims = sorted([slider_w, slider_h])
    short_dim, long_dim = dims[0], dims[1]
    
    if orig_h < orig_w: final_h, final_w = short_dim, long_dim
    else: final_w, final_h = short_dim, long_dim
    
    print(f"Target Resolution: {final_w}x{final_h}")

    pose_v = load_video_to_tensor(pose_path).permute(0,3,1,2) 
    pose_v = resize_for_rectangle_crop(pose_v, [final_h, final_w], "center")
    pose_v = (pose_v - 127.5) / 127.5
    
    img_t = load_image_to_tensor_chw_normalized(pil_img)
    img_t = resize_for_rectangle_crop(img_t, [final_h, final_w], "center")

    driving_v = None
    try:
        driving_v = load_video_to_tensor(drive_path).permute(0,3,1,2)
        driving_v = resize_for_rectangle_crop(driving_v, [final_h, final_w], "center")
        driving_v = (driving_v - 127.5) / 127.5
    except: pass

    smpl_v = pose_v
    if "smpl_downsample" in args.representation:
        smpl_v = F.interpolate(pose_v, scale_factor=0.5, mode='bilinear', align_corners=False)

    print("STEP 3/3: Running Sampling...")
    
    with torch.no_grad():
        p_in = pose_v.unsqueeze(0).to('cuda', torch.bfloat16)
        s_in = smpl_v.unsqueeze(0).to('cuda', torch.bfloat16)
        ori_i = img_t.unsqueeze(0).to('cuda', torch.bfloat16)
        
        img_lat = torch.concat([ori_i, torch.zeros_like(p_in[:, 1:])], dim=1)
        img_lat = model.encode_first_stage(rearrange(img_lat, 'b t c h w -> b c t h w').contiguous(), None, True).permute(0, 2, 1, 3, 4).contiguous()
        ref_lat = model.encode_first_stage(rearrange(ori_i, 'b t c h w -> b c t h w').contiguous(), None, True).permute(0, 2, 1, 3, 4).contiguous()
        
        if "smpl" in args.representation:
             pose_lat = model.encode_first_stage(rearrange(s_in, 'b t c h w -> b c t h w').contiguous(), None, True).permute(0, 2, 1, 3, 4).contiguous()
        else:
             pose_lat = model.encode_first_stage(rearrange(p_in, 'b t c h w -> b c t h w').contiguous(), None, True).permute(0, 2, 1, 3, 4).contiguous()

        value_dict = {'prompt': prompt, 'negative_prompt': "", 'num_frames': torch.tensor(pose_lat.shape[1]).unsqueeze(0), 'orig_height': final_h, 'orig_width': final_w, 'aesthetic_score': 6.0, 'negative_aesthetic_score': 2.5, 'fps': fps}
        model.conditioner.embedders[0].to('cuda')
        b, buc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, [1])
        c, uc = model.conditioner.get_unconditional_conditioning(b, batch_uc=buc)
        model.conditioner.embedders[0].cpu()

        for k in c: 
            if k != "crossattn": c[k], uc[k] = c[k][:1].to("cuda"), uc[k][:1].to("cuda")
        
        c.update({"concat_images": img_lat, "ref_concat": ref_lat, "concat_pose": pose_lat})
        uc.update({"concat_images": img_lat, "ref_concat": ref_lat, "concat_pose": pose_lat})
        if "smpl" in args.representation: c["concat_smpl_render"] = pose_lat; uc["concat_smpl_render"] = pose_lat

        if model.use_i2v_clip:
            model.i2v_clip.model.to('cuda')
            clip = model.i2v_clip.visual(ori_i.permute(0,2,1,3,4))
            c["image_clip_features"] = clip; uc["image_clip_features"] = clip
            model.i2v_clip.model.cpu()

        samples_z = model.sample(c, uc=uc, batch_size=1, shape=(pose_lat.shape[1], 16, final_h//8, final_w//8), ofs=torch.tensor([2.0]).to('cuda'), fps=torch.tensor([fps]).to('cuda'))
        
        print("Decoding...")
        samples_x = model.decode_first_stage(samples_z.permute(0, 2, 1, 3, 4).contiguous()).to(torch.float32)
        samples = torch.clamp((samples_x.permute(0, 2, 1, 3, 4).contiguous() + 1.0) / 2.0, 0, 1).cpu()

        final_vid_path = os.path.join(work_dir, "final_result.mp4")
        grid_vid_path = os.path.join(work_dir, "final_grid.mp4")
        
        # Save Clean
        result_vis = samples[0]
        frames_clean = [(255.0 * rearrange(f, "c h w -> h w c")).cpu().numpy().astype(np.uint8) for f in result_vis]
        with imageio.get_writer(final_vid_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None) as writer:
            for f in frames_clean: writer.append_data(f)

        # Save Grid
        pose_vis = torch.clamp((pose_v + 1.0) / 2.0, 0, 1).cpu()
        img_vis = torch.clamp((img_t + 1.0) / 2.0, 0, 1).cpu().repeat(pose_vis.shape[0], 1, 1, 1)
        stack_list = [pose_vis, img_vis]
        
        if driving_v is not None:
             driving_vis = torch.clamp((driving_v + 1.0) / 2.0, 0, 1).cpu()
             if driving_vis.shape[0] > result_vis.shape[0]: driving_vis = driving_vis[:result_vis.shape[0]]
             elif driving_vis.shape[0] < result_vis.shape[0]:
                 last = driving_vis[-1:]; pad = last.repeat(result_vis.shape[0] - driving_vis.shape[0], 1, 1, 1)
                 driving_vis = torch.cat([driving_vis, pad], dim=0)
             stack_list.append(driving_vis)
        
        stack_list.append(result_vis)
        final_video_tensor = torch.cat(stack_list, dim=3)
        
        frames_grid = [(255.0 * rearrange(f, "c h w -> h w c")).cpu().numpy().astype(np.uint8) for f in final_video_tensor]
        with imageio.get_writer(grid_vid_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None) as writer:
            for f in frames_grid: writer.append_data(f)
            
        print("✅ DONE!")
        return final_vid_path, grid_vid_path, pose_path

# --- B. Standalone Pose Extraction Logic (For the new tab) ---
def run_preprocessing_files(driving_file, ref_file, height, width, use_align, is_multi):
    # This uses a generator to update the logs real-time in the new tab
    if driving_file is None:
        yield "❌ Error: You must upload a driving video.", None
        return

    # 1. Create a clean temp folder for processing
    work_dir = tempfile.mkdtemp(prefix="scail_proc_")
    
    yield f"📦 Uploading files to server workspace: {work_dir}...\n", None
    
    # Copy inputs to temp dir
    driving_path = os.path.join(work_dir, "driving.mp4")
    shutil.copy2(driving_file, driving_path)
    
    ref_path = None
    if ref_file is not None:
        ref_path = os.path.join(work_dir, "ref.jpg")
        shutil.copy2(ref_file, ref_path)

    # 2. Construct Command
    script_name = "process_pose_multi.py" if is_multi else "process_pose.py"
    align_flag = "--use_align" if use_align else ""
    
    command = (
        f"cd SCAIL-Pose && "
        f"eval \"$(conda shell.bash hook)\" && "
        f"conda activate ./SCAIL_poses && "
        f"python NLFPoseExtract/{script_name} --subdir \"{work_dir}\" --resolution {height} {width} {align_flag}"
    )
    
    yield f"🚀 Executing Command:\n{command}\n\n", None
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, executable="/bin/bash")

    logs = ""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None: break
        if line: logs += line; yield logs, None

    # 3. Find Output
    output_candidates = ["rendered_aligned.mp4", "rendered.mp4", "pose.mp4"]
    final_output = None
    
    if process.returncode == 0:
        for fname in output_candidates:
            fpath = os.path.join(work_dir, fname)
            if os.path.exists(fpath):
                final_output = fpath
                break
        
        if final_output:
            # Create a dedicated subfolder in staging for this job to keep names clean
            staging_root = os.path.abspath("gradio_staging")
            job_id = str(uuid.uuid4())[:8]
            job_dir = os.path.join(staging_root, job_id)
            os.makedirs(job_dir, exist_ok=True)
            
            # Destination path
            staged_pose = os.path.join(job_dir, "rendered.mp4")
            shutil.copy2(final_output, staged_pose)
                
            msg = logs + f"\n\n✅ Success! File ready: {staged_pose}"
            yield msg, staged_pose
        else:
            yield logs + "\n\n⚠️ Script finished but output missing.", None
    else:
        yield logs + "\n\n❌ Failed.", None

# -----------------------------------------------------------------------------
# 5. UI
# -----------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Human: Pose to Video ")
    
    with gr.Tabs():
        # --- TAB 1: The original App ---
        with gr.TabItem("Generate Video (Unified)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Inputs")
                    prompt_in = gr.Textbox(label="Prompt", value="A person dancing...", lines=3)
                    drive_in = gr.Video(label="Driving Video (Motion Source)", format="mp4")
                    ref_in = gr.Image(label="Reference Image (Character)", type="filepath")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            gw = gr.Slider(256, 1024, 896, step=64, label="Target Width"); gh = gr.Slider(256, 1024, 512, step=64, label="Target Height")
                        steps = gr.Slider(10, 100, 50, label="Steps"); cfg = gr.Slider(1, 10, 4, label="CFG"); fps = gr.Slider(8, 30, 16, label="FPS")
                        seed = gr.Number(-1, label="Seed")
                        with gr.Row():
                            align_chk = gr.Checkbox(label="Align Pose"); multi_chk = gr.Checkbox(label="Multi Person")
                    
                    btn_run = gr.Button("🚀 Generate Video", variant="primary")
                    logs_out = gr.Textbox(label="Live Logs", lines=10, max_lines=10, interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 2. Results")
                    out_final = gr.Video(label="Final Generated Video")
                    out_grid = gr.Video(label="Comparison Grid")
                    out_pose = gr.Video(label="Extracted Pose (Intermediate)")

            btn_run.click(
                fn=run_scail_pipeline,
                inputs=[prompt_in, drive_in, ref_in, seed, steps, cfg, fps, gw, gh, align_chk, multi_chk],
                outputs=[out_final, out_grid, out_pose],
                concurrency_limit=1
            )
            
            timer = gr.Timer(0.5)
            timer.tick(read_logs, outputs=logs_out)

        # --- TAB 2: The New Feature (Pose Extract Only) ---
        with gr.TabItem("Extract Pose Only"):
            gr.Markdown("### Upload files to extract pose")
            gr.Markdown("Useful if you just want to get the skeleton animation without running the AI Generator.")

            with gr.Row():
                with gr.Column():
                    in_drive = gr.Video(label="Upload Driving Video (Required)", format="mp4")
                    in_ref = gr.Image(label="Upload Ref Image (Optional)", type="filepath")
                    
                    with gr.Row():
                        h_in = gr.Number(label="Height", value=512); w_in = gr.Number(label="Width", value=896)
                    with gr.Row():
                        align_in = gr.Checkbox(label="Align Pose"); multi_in = gr.Checkbox(label="Multi Person")
                    
                    btn_p = gr.Button("Extract Pose", variant="primary")
                
                with gr.Column():
                    log_p = gr.Textbox(label="Status/Logs", lines=15)
                    out_pose_down = gr.File(label="Download Pose Video")
                    # Visual check
                    out_pose_vis = gr.Video(label="Preview", interactive=False)

            # Wire up the new feature
            btn_p.click(
                fn=run_preprocessing_files,
                inputs=[in_drive, in_ref, h_in, w_in, align_in, multi_in], 
                outputs=[log_p, out_pose_vis] # The video output also works as a file download usually
            )

demo.queue()
# Allow both output directories so files can be served
allowed_dirs = [os.path.abspath("gradio_outputs"), os.path.abspath("gradio_staging")]
os.makedirs(allowed_dirs[0], exist_ok=True)
os.makedirs(allowed_dirs[1], exist_ok=True)

demo.launch(server_name="0.0.0.0", server_port=4455, allowed_paths=allowed_dirs, theme=gr.themes.Soft())
