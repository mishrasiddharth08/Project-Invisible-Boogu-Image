"""Explicit trunk policy, validated before allocating models."""
import math

def validate_task(trunk, is_img2img, reference_count=0):
    if trunk not in ("base", "turbo", "edit", "edit-turbo"):
        raise ValueError("Unknown Boogu model mode. Select a supported Boogu checkpoint.")
    if "edit" in trunk:
        if not is_img2img:
            raise ValueError("For editing, open img2img and upload one image. Edit models cannot run in txt2img.")
        if reference_count != 1:
            raise ValueError("Upload exactly one image in img2img for Boogu editing.")
    elif is_img2img or reference_count:
        raise ValueError("For editing in img2img, select Boogu Edit or Edit-Turbo. Use Base/Turbo in txt2img.")

def task_hint(trunk, is_img2img):
    editing = "edit" in trunk
    if editing != bool(is_img2img):
        return ("**Open img2img to edit.** Upload one image, describe the change, then Generate."
                if editing else "**Select Boogu Edit or Edit-Turbo to edit here.** Base/Turbo create new images in txt2img.")
    action = "Upload one image and describe what to change." if editing else "Describe the image you want."
    speed = "Steps slider is adjustable; 4 is recommended for Turbo. CFG stays 1." if "turbo" in trunk else "Steps slider is adjustable; start with 28 steps and CFG 4."
    return f"{action} Press **Generate**. {speed}"

def profile(vram_gb, trunk):
    # These are requested resolution ceilings, NOT benchmark guarantees.
    if 0 < vram_gb < 4:
        precision, cap = "bf16/int8_convrot", 512
    elif 0 < vram_gb < 8:
        precision, cap = "bf16/int8_convrot", 768
    elif vram_gb <= 8:
        precision, cap = "nvfp4/int8_convrot", 1024
    elif vram_gb <= 12:
        precision, cap = "fp8_scaled", 1024
    elif vram_gb <= 16:
        precision, cap = "fp8_scaled", 1536
    elif vram_gb < 24:
        precision, cap = "bf16/fp8_scaled", 1536
    else:
        precision, cap = "bf16", 2048
    if "turbo" in trunk:
        cap = min(cap, 1024)
    # Official sequential offload is the conservative default for ALL profiles.
    return {"precision": precision, "cap": cap, "default": min(1024, cap), "offload": "auto"}

def settings(trunk, width, height, ceiling=1024, vram_gb=0, distill=None, steps=None, cfg=4):
    turbo = "turbo" in trunk
    steps = (4 if turbo else 28) if steps is None else int(steps)
    if steps < 1:
        raise ValueError("Choose at least 1 step using the Steps slider.")
    if distill is not None and bool(distill) != turbo:
        raise ValueError("Distill must be on for a merged Turbo trunk and off for Base/Edit. Switch checkpoint to change it.")
    cap = min(int(ceiling), profile(vram_gb, trunk)["cap"])
    width, height = max(256, int(width)), max(256, int(height))
    scale = min(1., cap / max(width, height))
    width, height = max(256, math.floor(width * scale / 16) * 16), max(256, math.floor(height * scale / 16) * 16)
    return dict(width=width, height=height, num_inference_steps=steps,
                text_guidance_scale=1.0 if turbo else max(2., min(5., float(cfg))))

def arguments(trunk, prompt, negative, resolved, reference=None):
    edit, turbo = "edit" in trunk, "turbo" in trunk
    if edit != (reference is not None):
        raise ValueError("Edit/Edit-Turbo require one img2img image; Base/Turbo only support txt2img.")
    args = dict(instruction=[prompt], negative_instruction=negative, image_guidance_scale=1.0, **resolved)
    args.update(max_input_image_pixels=resolved["width"] * resolved["height"],
                max_input_image_side_length=2 * max(resolved["width"], resolved["height"]))
    if turbo:
        args.update(empty_instruction_guidance_scale=0.0, use_dmd_student_inference=True,
                    dmd_conditioning_sigma=0.0 if edit else 0.001)
    if edit:
        args.update(input_images=[[reference]], align_res=False)
    return args
