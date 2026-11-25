# PROJECT INVISIBLE — Boogu-Image-0.1


## Install from GitHub

In Forge, open **Extensions > Install from URL**, paste the repository URL below, and click **Install**. Restart Forge afterwards.

```text
https://github.com/mishrasiddharth08/Project-Invisible-Boogu-Image
```

Alternatively, download **Code > Download ZIP**, extract it into `Forge/extensions/`, and name the folder `project-invisible-boogu-image`.

Model weights are separate. Follow **MANUAL-DOWNLOADS.md** before generating. This source release excludes private machine settings and local validation images/reports. The benchmark descriptions below describe earlier local tests; they are not guarantees for your hardware. Adapter trunk permissions must be configured for your own adapters.

## Start here

1. Close Forge. This folder belongs in `Forge/extensions/project-invisible-boogu-image`.
2. Start Forge normally. Choose **boogu** in the existing **UI Preset** dropdown, then select a **Boogu — ...** checkpoint in the normal checkpoint dropdown.
3. **New image:** choose Base or Turbo, open **txt2img**, type a description, then **Generate**. **Edit an image:** choose Edit or Edit-Turbo, open **img2img**, upload one image, describe the change, then **Generate**. All editing, including inpainting, stays inside img2img.
4. Need models? **Manual download is recommended:** follow [MANUAL-DOWNLOADS.md](MANUAL-DOWNLOADS.md). Or double-click **setup-models.cmd**, choose automatic download and select the models you want from its numbered list. It scans existing files first and reuses them. Type **DOWNLOAD** only if you approve the listed missing files. Anything else cancels.
5. Start Forge again. The same files are reused on future launches. Startup and Generate never download anything.

This is an integration implementation, not a claim that all GPU acceptance tests passed. See `VALIDATION.md` for the actual checks and outstanding gates.

## Faster generation and live progress

### GPU compatibility

This is **not a promise of support for every GPU or every VRAM size**. The physical generation tests were performed on an RTX 5090. Drivers, the existing Forge/PyTorch build, working-tensor size and available system RAM still matter.

- **NVIDIA CUDA:** uses Forge's current GPU instead of fixing generation to GPU 0. Native BF16 is preferred. If native BF16 is unavailable, an experimental FP32 path uses unquantized weights and the official PyTorch attention/norm implementations. FP32 needs more RAM and is slower; older NVIDIA hardware has not been physically tested.
- **AMD HIP/ROCm:** experimental compatibility path, only when the existing Forge/PyTorch installation already exposes the AMD GPU. It avoids the optional NVIDIA fused operations and requires unquantized Boogu and Qwen3-VL weights. BF16 is used when available, otherwise FP32. No AMD GPU was available for validation.
- **DirectML and CPU-only installations:** unsupported by this extension. No drivers, replacement Torch build, extra environment or model weights are installed automatically.
- **Packed weights:** FP8/INT8/NVFP4 kernel compatibility is not universal. The AMD/legacy compatibility path rejects packed files by inspecting their headers, even if a filename says BF16. It reuses an existing unquantized encoder when one is available.

Below 4 GiB total VRAM, the resolution cap/default becomes 512; from 4 to below 8 GiB it becomes 768. At 8 GiB and above the default remains 1024. These are conservative policies, **not measured minimum-VRAM guarantees**. Offloading cannot make the largest working tensors fit an arbitrarily small GPU. If sequential offload still fails, reduce the prompt/resolution or use hardware with more available memory.

After installing this update, **close and restart Forge completely**. A browser refresh cannot reload the Python extension.

For this RTX 5090, choose **boogu_image_turbo_hotfix_int8_convrot** at **1024 x 1024**. Turbo starts at four steps and CFG 1; use the Steps slider to change the step count. The preset now prefers an existing INT8 hotfix on first selection; it preserves a checkpoint you already selected.

The native progress display shows loading, prompt encoding, denoising and image decoding. Enable live previews in Forge Settings. Boogu checks progress every second; a new preview needs a completed generation step, so slow steps take longer. Previews are rough, lightweight approximations. Judge sharpness from the saved PNG, which uses the full official image decoder.

Repeated prompts are faster again: the extension keeps up to four exact encoder results in **at most 64 MiB of system RAM**, with no persistent GPU cache or files written to disk. Changing prompt text, reference pixels or encoding settings causes a fresh encode. Switching checkpoint, encoder or LoRA configuration clears the cache. Set `"prompt_cache_mb": 0` in config.json and restart Forge to disable it.

Editing now also seeds the official reference-image VAE sampling. Reusing the same reference, prompt, settings and seed is reproducible in the tested configuration; the wrapper restores the random state afterward.

With this cache, repeated short-prompt Turbo generations measured **6.5–6.7 seconds** on the same RTX 5090, versus **9.84 seconds** with the cache disabled in the same process. An 879-token prompt measured **6.85–6.99 seconds** cached versus **9.95 seconds** uncached. Output pixels were identical in both comparisons; the caches used about 0.4 MB and 7.2 MB of system RAM respectively. A new prompt still needs encoding, and the first generation after startup still pays initialization costs.

Measured on an RTX 5090, short prompt, 1K, seed 1234, existing FP8 Qwen3-VL encoder, with live previews enabled: warm INT8 Turbo took **9.58 seconds** in auto/model mode versus **13.05 seconds** in sequential mode (27% less time). Peak allocated VRAM was **11.51 GiB** versus **2.41 GiB**. The two final images were pixel-identical. BF16 Turbo took **11.73 seconds**, with **19.90 GiB** peak allocated VRAM. These are end-to-end warm inference times, excluding initial model construction. The first generation is slower; long prompts and editing cost more. Reports are in validation/performance-report*.json.

## The four checkpoints

| Checkpoint | Where to use it | Steps / CFG | Default size |
|---|---|---|---|
| Base | txt2img | 28 / 4 by default; steps adjustable, CFG 2–5 | 1024 |
| Turbo | txt2img | 4 / 1 by default; steps adjustable | 1024 |
| Edit | img2img or inpaint, one reference | 28 / 4 by default; steps adjustable, CFG 2–5 | 1024 |
| Edit-Turbo | img2img or inpaint, one reference | 4 / 1 by default; steps adjustable | 1024 |

Edit-Turbo always uses **BooguImageTurboPipeline**, even when its downloaded model index says BooguImagePipeline. The implementation sets the official DMD flags and sigma 0.0 for Edit-Turbo / 0.001 for Turbo. It does not route Boogu through an SD UNet or Flux sampler.

Newer `turbo_hotfix` and `edit_turbo_hotfix_1k_20260708` files appear ahead of old files. Setup requests these hotfixes. An explicitly selected older checkpoint is still the one used.

The Boogu panel appears only for Boogu checkpoints. It tells you what to do in the current tab. Fast mode is automatic: Turbo defaults to four steps and keeps CFG 1. The normal Steps slider controls the actual step count for every Boogu model. The optional image-size section limits memory use; no distillation switch needs adjusting. The extension never silently switches models.

## If an image looks slightly blurry

For fine repeating grid patterns, enable **Boogu → Optional settings → Reduce fine grid patterns (DeGrid)**. This now uses the Apache-2.0 [ComfyUI-DeGrid](https://github.com/lunaaispace-eng/ComfyUI-DeGrid) core pinned at `5699bc33f71e1be12523fdea105cc9c2abfe1cd0`, with automatic correction strength and skip-when-clean detection. It runs on CPU in both Boogu tabs, before inpaint compositing, and stays off by default. Detection is imperfect: real fine detail can still change. It is not a sharpening filter. The saved image records `DeGrid auto` when enabled. No ComfyUI install, extra model downloads or Forge core changes are needed. The earlier seven-tap implementation remains as reference code; generation uses DeGrid's nine-tap filter. Source attribution and the upstream license are in `pi_boogu_image/lib/vendor/degrid/`.

Open the saved PNG at 100% size before judging it. In the inspected examples, the prompts requested film grain, strong background blur or smoothing; no broken Boogu decoder was found. Try [this clear-detail portrait prompt](examples/sharp-portrait.txt), keep the subject in focus, and remove conflicting requests for blur, grain and airbrushed skin. Keep 1024 as the starting size on suitable hardware. Changing the prompt changes the picture; it is not a guaranteed sharpening filter. Four steps remain the recommended Turbo starting point; more steps are slower and do not guarantee better quality. Original and revised same-seed examples are in `validation/quality-original.png` and `validation/quality-clear-detail.png`.

Base/Turbo in img2img and Edit without a reference return a message and no image. Inpainting runs the official single-reference edit and composites the result through Forge's mask; this is not a separate native masked-denoising model. Hires fix is rejected. Forge samplers, SD denoise settings and ControlNet are not Boogu inference controls.

## Existing models and folders

The extension searches all these folders before offering a download:

- `models/Boogu/`
- `models/Stable-diffusion/`
- `models/diffusion_models/`
- `models/text_encoder/` and `models/text_encoders/`
- `models/VAE/`
- `models/Lora/`
- Additional paths in `config.json`.

Keep a real Boogu transformer safetensors file, a Qwen3-VL **8B** encoder with its vision weights, and a **Flux1 16-channel VAE**. Flux2 VAEs and Qwen3 text-only encoders are rejected. Do not rename an unrelated model to Boogu. Detection verifies tensor headers and required keys. Incomplete `.partial` files are ignored.

Single-file transformer weights use the pinned official transformer config and strict key/shape matching. The files bundled with this extension are not a complete processor. Single-file installs can reuse a complete local processor or the revision-pinned Hugging Face cache. If neither is available, use `setup-models.cmd` to list and fetch the missing files, or follow [MANUAL-DOWNLOADS.md](MANUAL-DOWNLOADS.md) to put the full pinned processor in `models/Boogu/support/processor/`. Complete local official pipeline directories containing `model_index.json` are also discovered. GGUF is deliberately not registered: no proven Boogu GGUF pipeline loader is bundled.

## Memory

| VRAM | Requested precision preference | Resolution ceiling |
|---|---|---|
| Below 4 GB | Compatible quantized / BF16 storage | 512 |
| 4 to below 8 GB | Compatible quantized / BF16 storage | 768 |
| 8 GB | NVFP4 / INT8 ConvRot when kernels support it | 1024 |
| Over 8–12 GB | FP8 scaled | 1024 |
| Over 12–16 GB | FP8 scaled | 1536 |
| Over 16–under 24 GB | BF16 / FP8 scaled | 1536 |
| 24 GB or more | BF16 | 2048 for Base/Edit |

The default is 512 below 4 GiB, 768 below 8 GiB, and 1024 otherwise; Turbo never exceeds 1024. These are policy ceilings, not guaranteed OOM-free profiles. The default `"offload": "auto"` selects official model CPU offload when the largest component fits in currently free VRAM with a resolution-dependent working-memory reserve. Otherwise it selects official sequential CPU offload. Model offload keeps each component on the GPU while that component runs; sequential offload transfers individual layers and uses less VRAM but is slower. VAE tiling and slicing remain enabled. If model offload runs out of VRAM, generation retries once with sequential offload and the same seed. Set `"offload": "sequential"` in config.json for minimum VRAM usage. Restart Forge after changing it. CPU RAM, prompt length, reference size and other GPU users also affect memory and speed.

FP8 weights retain their scales. INT8 ConvRot and NVFP4 use the host's native Comfy-Kitchen layouts, not guessed conversions. Packed linear weights are unpacked one layer at a time for official pipeline computation. This saves weight storage but is slower than dedicated quantized matrix multiplication. Missing or unsupported kernels fail with a message. Applying a LoRA materializes only its targeted quantized transformer layers on CPU for official PEFT loading, increasing RAM use for those layers. LoKr keeps the original packed layers and applies a separate adapter branch. No fake SD `medvram` is enabled.

## LoRA and LoKr

Put compatible Boogu LoRA or LoKr `.safetensors` files in Forge's normal `models/Lora` folder, refresh Extra Networks, and select the adapter. Both types use `<lora:name:0.8>` in the positive prompt; there is no separate LoKr tab or tag. Start at strength 0.8. Remove the tag to stop using it. LoKr does not mean any adapter from another model will work. A LoRA must declare its Boogu trunk through `boogu_trunk` / `ss_boogu_trunk` metadata or an unambiguous Boogu filename. `ss_base_model=Boogu` alone does not identify which trunk is safe.

Base, Turbo, Edit and Edit-Turbo are hard-separated. Foreign cards are greyed out with “not a Boogu LoRA” when Boogu is selected, and return to normal when another engine is selected. Generation enforces the same rule independently. Turbo rank-128 distillation LoRAs are rejected on already merged Turbo checkpoints and are never applied to Base/Edit. Adapters and cached pipeline are released on checkpoint switches. LoRA schedules, block weights, and other Extra Networks types are rejected.

## Missing Python packages

Do not create another environment. `install.py` only reports missing dependencies; it never changes Forge's Torch/CUDA or downloads models. To install the extension requirements yourself, close Forge, open a terminal in this extension folder and run:

```bat
"..\..\venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Review pip's proposed changes before proceeding if this Forge has custom dependency pins. The upstream project declares Python >=3.10,<3.13 and Torch >=2.7.1,<2.12. This host has Python 3.13 / Torch 2.13, outside that upstream support range. The vendored source avoids installing an incompatible upstream package or replacing Forge's Torch; this does **not** establish full runtime compatibility. A dependency/import failure is a setup error, not a generated image.

To choose a different confirmed download, open a terminal in this folder:

```bat
"..\..\venv\Scripts\python.exe" -m pi_boogu_image.download.setup --trunk edit-turbo --precision int8_convrot
```

Use `--plan` to list files without downloading. No API keys are stored in this extension. Downloads use pinned public Hugging Face revisions and verify safetensors headers after completion.

## Removal and coexistence

Close Forge and remove this extension folder. Restart Forge. No Forge core files were changed. Models remain in their model folders. Other PROJECT INVISIBLE extensions keep their own packages and caches. The extension chains the existing Generate and processing hooks and passes unrelated checkpoints through unchanged. Unknown Neo hook signatures disable the integration with a console message. It adds only its own `boogu` preset choice and options; it does not change preset enum members, other presets, global Torch operations, attention processors, or samplers.

## Copies and sources

Install inside your own Forge `extensions` folder. Keep any backup outside that folder.

`sources.json` records pinned source and model revisions. Apache-2.0 license and upstream attribution are included. Official Boogu code is preserved under `pi_boogu_image/lib/vendor/boogu/`.

- Official inference: https://github.com/Boogu-Project/Boogu-Image
- Native Comfy-Org exports: https://huggingface.co/Comfy-Org/Boogu-Image
- Official model family: https://huggingface.co/Boogu
- Host: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo

Live progress: Boogu checks for progress every second. With live previews enabled, a new preview is shown after a completed generation step, at most once per second. Slow steps can take longer; loading and prompt encoding do not yet have an image. Other checkpoints keep their previous refresh setting. Restart Forge and refresh the browser after updating.

Boogu shows two progress bars: Current image (sampling steps) and Overall (all images in the request). They refresh every second while generation runs. Switching checkpoints directly releases Boogu models, prompt cache and available GPU allocations. If a generation is running, release waits for it to finish safely. Forge, drivers and other extensions retain their own memory; usage will not fall to zero.


LoRA uses the official Boogu loader with PEFT. PEFT 0.18.0 was installed into this Forge environment without replacing Torch. LoKr supports validated transformer linear-layer formats; unsupported adapter layouts fail with a message instead of silently changing the model. Restart Forge after this update.

Supported adapter scope: transformer Linear LoRA with A/B tensor pairs, and LyCORIS Linear LoKr with full or factorized W1/W2 tensors. LoKr convolution/Tucker variants, DoRA/LoHa and hybrid LoRA+LoKr inside a single file are rejected. Separate compatible LoRA and LoKr files can be combined. Strength zero disables that adapter's contribution. Adapter names must identify the exact Base, Turbo, Edit or Edit-Turbo model family. We validated small test adapters; a trained third-party Boogu adapter still needs its own image-quality check.


The EVA Boogu adapter is explicitly approved by the user for all four Boogu model types via config.json adapter_trunks. AI Toolkit ss_base_model_version metadata and diffusion_model-prefixed LoKr layers are recognised. Other adapters retain their compatibility checks. Cross-checkpoint visual results are not guaranteed.



## Special Thanks

- [**r/sdforall**](https://www.reddit.com/r/sdforall/) - community discussion and testing
- [**r/SECourses**](https://www.reddit.com/r/SECourses/) - community discussion and testing
- [**r/malcolmrey**](https://www.reddit.com/r/malcolmrey/) - community discussion and testing
- [**Haoming02 / sd-webui-forge-classic (neo branch)**](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) - the Forge Neo tree this extension targets
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) - reference for upstream sampler/scheduler coverage
- The Forge / AUTOMATIC1111 community - for the extension ecosystem this plugs into
