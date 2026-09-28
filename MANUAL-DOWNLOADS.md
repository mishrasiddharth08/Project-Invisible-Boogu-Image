# Manual Boogu model download (recommended)

Manual download lets you review every large file yourself. These repositories and filenames come from this extension's pinned `sources.json` and downloader manifest.

## 1. Choose the model type

| Model | Use in Forge | Normal behavior |
|---|---|---|
| Base | txt2img | Standard model, 25–50 steps |
| Turbo | txt2img | Faster, fixed at 4 steps |
| Edit | img2img or inpaint with one reference image | Standard model, 25–50 steps |
| Edit-Turbo | img2img or inpaint with one reference image | Faster, fixed at 4 steps |

## 2. Download one transformer

Open the pinned [Comfy-Org/Boogu-Image repository](https://huggingface.co/Comfy-Org/Boogu-Image/tree/0e060b5c47ce43918f5943d166e57b73b09d72f8/diffusion_models), choose one file below, and place it in:

`Forge/models/Boogu/diffusion_models/`

| Model | Available files |
|---|---|
| Base | `boogu_image_base_bf16.safetensors`, `boogu_image_base_fp8_scaled.safetensors`, `boogu_image_base_nvfp4.safetensors` |
| Turbo | `boogu_image_turbo_hotfix_bf16.safetensors`, `boogu_image_turbo_fp8_scaled.safetensors`, `boogu_image_turbo_hotfix_int8_convrot.safetensors`, `boogu_image_turbo_hotfix_nvfp4.safetensors` |
| Edit | `boogu_image_edit_bf16.safetensors`, `boogu_image_edit_fp8_scaled.safetensors`, `boogu_image_edit_int8_convrot.safetensors`, `boogu_image_edit_nvfp4.safetensors` |
| Edit-Turbo | `boogu_image_edit_turbo_hotfix_1k_20260708_bf16.safetensors`, `boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors` |

Packed FP8, INT8 and NVFP4 files require compatible GPU support. BF16 transformer files are the portable unquantized choice.

## 3. Download the shared encoder and VAE

From the same verified repository, download:

- `text_encoders/qwen3vl_8b_fp8_scaled.safetensors` to `Forge/models/Boogu/text_encoders/`
- `vae/flux1_vae_bf16.safetensors` to `Forge/models/Boogu/vae/`

The manifest offers only the FP8 shared Qwen3-VL 8B encoder. For AMD or another path requiring unquantized weights, reuse an existing supported unquantized **Qwen3-VL 8B** encoder with vision weights. This extension does not claim a verified BF16 encoder download filename or link.

The processor/tokenizer files are already bundled with this extension. If that bundle is missing, the automatic planner lists the exact processor files from the pinned [Boogu/Boogu-Image-0.1-Base repository](https://huggingface.co/Boogu/Boogu-Image-0.1-Base/tree/334ad7e5c6343d33308b306127e047b133e86926/processor) for `Forge/models/Boogu/support/processor/`.

## 4. Verify without downloading

Run this from the extension folder to scan existing files and print what is still missing:

```bat
setup-models.cmd --trunk turbo --precision int8_convrot --plan
```

Change the trunk and precision to match your chosen file. `--plan` never downloads and never asks for model selection. Existing Forge model folders and additional paths from `config.json` are scanned first.
