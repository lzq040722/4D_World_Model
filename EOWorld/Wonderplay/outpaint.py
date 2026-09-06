import os
import torch
from PIL import Image
from diffusers import QwenImageEditInpaintPipeline


model_path = "/root/autodl-tmp/huggingface/hub"

pipeline = QwenImageEditInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("pipeline loaded")

pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)


# 原图
image = Image.open(
    "3d_result/wonderplay/venice/Gen-02-09_21-09-32/multiview_000/generation_condition.png"
).convert("RGB")

mask = Image.open(
    "3d_result/wonderplay/venice/Gen-02-09_21-09-32/multiview_000/generation_hole_mask.png"
).convert("L")


prompt = (
    "Complete the white masked missing area only. Treat all white pixels as empty unknown content that must be "
    "replaced, not preserved. Continue the same canal-side scene visible on the right: ordinary mid-rise Venetian "
    "waterfront buildings, orange and yellow plaster facades, rows of windows, canal water, and cloudy blue sky. "
    "Match the existing perspective, camera position, building scale, water level, lighting, and color temperature. "
    "Use the visible input image as geometry and composition reference only, and do not copy its blurry "
    "3D-reconstruction artifacts. Make the newly generated area clean and sharp, with crisp facade edges, readable "
    "window shapes, natural water ripples, and consistent reflections. Do not introduce a new landmark or focal object. "
    "Keep the unmasked right-side content unchanged."
)

inputs = {
    "image": image,
    "mask_image": mask,
    "prompt": prompt,
    "generator": torch.Generator(device="cuda").manual_seed(0),
    "height" : 512,
    "width": 512,

    "true_cfg_scale": 4.0,
    "negative_prompt": (
        "blurry, soft focus, low detail, smeared texture, painterly, 3D reconstruction artifacts, "
        "warped architecture, distorted windows, broken perspective, inconsistent reflections, "
        "blank white area, unfilled mask, white rectangle, visible seam, hard mask boundary, "
        "cathedral, basilica, church dome, landmark dome, tower, crane, large boat, "
        "color shift, over-smoothed, noise, watermark, text"
    ),

    "num_inference_steps": 40,

    # 这类大面积白色空洞需要 1.0；低于 1.0 容易把白块照抄回来。
    "strength": 1.0,

    "guidance_scale": 3.0,
    "num_images_per_prompt": 1,
}


with torch.inference_mode():
    output = pipeline(**inputs)

output_image = output.images[0]

output_image.save("Temp/outpaint.png")
