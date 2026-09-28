# Validation — September 20, 2026

## Real image generation

The pinned official pipeline generated and saved these 1024×1024 images using existing local files, the existing Forge Python, and an RTX 5090. No model weights were downloaded.

| Run | Checkpoint | Steps / CFG | Peak allocated | Peak reserved | Result |
|---|---|---|---|---|---|
| Base | Base BF16 | 28 / 4 | 2.413 GiB | 3.170 GiB | Red teapot image generated |
| Turbo | Turbo hotfix INT8 ConvRot | 4 / 1 | 2.411 GiB | 3.166 GiB | Red teapot image generated |
| Edit-Turbo | Legacy Edit-Turbo INT8 ConvRot | 4 / 1 | 2.413 GiB | 2.559 GiB | Single reference edited from red to blue |

All used Qwen3-VL-8B FP8 scaled, Flux1 BF16 VAE, sequential CPU offload, and seed 1234. Images and machine-readable reports are in `validation/`. The Edit-Turbo hotfix was not present locally; the explicit legacy checkpoint was tested. Setup prefers the July 8, 2026 hotfix when downloads are approved.

## Automated checks

- 33 tests: trunk routing, references, fixed Turbo defaults, distillation locks, profiles, malformed files, strict tensor loading, FP8 scales, native NVFP4 round-trip, native LoRA classification, confirmation-only downloads, reuse, hotfix preference, hook passthrough, existing callback chaining, and hook-drift rollback.
- Official transformer and Qwen3-VL exported parameter keys match their instantiated official model classes: no missing or extra keys.
- Actual Gradio 4.40.0 controls construct successfully; both Boogu accordions start hidden for an unrelated checkpoint.
- Existing local Turbo checkpoint, encoder, VAE and bundled tokenizer produce an empty download plan.
- All extension Python files compile.

## Live Forge Neo verification

- Started the existing Forge Python with all existing enabled engines, separate test settings, no installs and offline model access, on localhost port 7871.
- Confirmed **boogu** appears in the native **UI Preset** dropdown alongside Ideogram, FIBO, SenseNova and the stock presets.
- Selected boogu: it selected a Turbo hotfix checkpoint, displayed the Boogu-only accordion, and set the stock controls to 4 steps / CFG 1 / 1024×1024.
- Clicked the stock **Generate** button: the gallery received a 1024×1024 image with `Sampler: Boogu official DMD`, `Steps: 4`, `CFG scale: 1.0`.
- Switched to Ideogram: its original checkpoint and controls returned, the Boogu accordion disappeared, and the count of disabled Boogu-marked LoRA cards returned to zero.
- Switched back to boogu successfully. The browser is left on this preset in the test session.
- Called the actual `/sdapi/v1/img2img` route with an Edit-Turbo override and one reference. It returned one image at 4 steps / CFG 1 despite requested 28 steps / CFG 4.
- Called that route with a Turbo checkpoint and the same reference: it returned zero images and an explanatory message, without an SD/Flux generation fallback.
- Restarted the test server repeatedly with local assets and offline Hub mode; the four checkpoint entries remained available without re-downloading weights.

The live check caught and fixed Neo's late quicksettings construction and Gradio's transient component updates. The Gradio construction probe now explicitly verifies late binding of both panels.

## Remaining verification

Non-Turbo Edit weights, Edit-Turbo July hotfix weights, full-model NVFP4 images, real compatible LoRA application, and other GPU/VRAM classes have not been benchmarked. PEFT is absent in this host until the requirements are installed. The NVFP4 numerical test is a small native-kernel round-trip, not a full Boogu image. The runtime hooks are tested for unchanged forwarding to other engines; an unrelated model's full image output was not regenerated for comparison.

No universal no-OOM guarantee is made. The measured figures above are this GPU, these prompts and resolutions. The upstream package support range excludes this host's Python 3.13 / Torch 2.13; the recorded successful source-based runs do not prove every upstream feature works.

## Performance and preview update

Auto/model offload and sequential offload were compared with the same Turbo INT8 hotfix, prompt, seed 1234, 1024 resolution and four steps. Both emitted four real latent previews and pixel-identical final images. Warm wall times: 9.575 s versus 13.047 s. Peak allocated/reserved GiB: 11.505/12.027 versus 2.411/2.537. The first auto run also paid process/kernel initialization, so cold runs are not used to claim a speedup. BF16 Turbo auto/model mode also generated successfully: warm 11.728 s, 19.898/20.666 GiB allocated/reserved, four previews. The existing process on port 7860 was not stopped. No model downloads or dependency changes were performed.

Automated additions cover conservative memory selection, cancellation and restoration of temporary pipeline observers. Sequential fallback after an actual CUDA OOM has not been deliberately forced.

Native Generate on the isolated Forge server produced a 1K Turbo image. The native progress endpoint delivered intermediate previews at steps 1, 2 and 3, with 26%, 51% and 76% progress; stage text showed loading, prompt encoding, denoising, decoding and saving. The browser gallery showed the completed image. See validation/progress-report.json. Edit-Turbo auto/model mode also passed: warm 12.555 s, 12.223 GiB allocated, four previews.

## Bounded encoder cache and recovery validation

The encoder cache stores exact official outputs on CPU, with a 64 MiB / four-entry bound. Tests cover changed text, changed reference pixels, changed settings, mutation isolation, eviction, disabled caching and prompt-cache lifetime. Simulated CUDA allocation failures verify that the old pipeline is released before retry, the original seed is restored and at most one sequential retry occurs. No actual GPU exhaustion was deliberately induced. A reference-RNG test verifies seeded editing and restoration of global random state. All 33 tests passed.

| Test | Warm cached seconds | Warm uncached seconds | Cache RAM bytes | Output identical |
|---|---|---|---|---|
| Turbo short prompt | 6.687, 6.521 | 9.835 | 385400 | Yes |
| Turbo 879-token prompt | 6.852, 6.994 | 9.948 | 7207800 | Yes |
| Legacy Edit-Turbo single reference | 8.799, 9.062 | 11.952 | 1869600 | Yes |

Base BF16, 28 steps / CFG 4: 34.940 s warm, 19.900 GiB peak allocated, 28 previews. First and cached results are pixel-identical. Cold initialization remains slower. These are single-machine measurements, not cross-GPU performance guarantees.

The earlier performance-report-edit-cached.json exposed nondeterministic reference VAE sampling in the upstream edit path, even between cache hits. The wrapper now scopes CPU/current-CUDA random state to the supplied seed for editing, then restores it. The seeded regression report above confirms pixel-identical editing with and without the encoder cache. Vendored official sampling code is unchanged.

Final installed-build Forge API checks passed: first 1K Turbo request 16.475 s, repeated request 6.827 s, one image each and intermediate previews delivered. Infotext correctly reports model offload. Edit-Turbo returned one image at four steps / CFG 1; Turbo img2img returned zero images with the mode explanation. See progress-report-final.json, progress-report-warm-final.json and forge-api-report.json in validation/.

## Hardware compatibility update

40 tests passed. Added tests cover current GPU selection, mocked HIP detection, FP32 fallback without native BF16, unsupported runtimes, packed-header rejection on the portable path, 512/768 small-VRAM limits, and actual BF16-to-FP32 tensor loading.

The portable official PyTorch branch generated a 512x512 Turbo image on the available RTX 5090 with unquantized BF16 transformer and encoder, sequential CPU offload, and four DMD steps. Peak allocated VRAM was 1.160 GiB; inference took 94.344 seconds in that cold validation. This demonstrates the code path on NVIDIA, not AMD hardware compatibility or a guarantee on a physically small GPU. The initial sandboxed probe exited without a report; the subsequent authorized full probe passed. See hardware-report.json and portable-512-validation.png.

AMD HIP/ROCm, legacy NVIDIA FP32 full-model inference, DirectML, and arbitrarily small VRAM are not claimed as validated. DirectML/CPU-only builds are explicitly unsupported. AMD/legacy compatibility mode requires existing unquantized weights; it does not install drivers, change the Forge Torch build, or download anything.

## Simple UI and download choices

49 unittest checks passed, including construction of both real Gradio panels, repeated/transient component callbacks, strict model/tab/reference routing, automatic fast mode, manual download without network, model-list selection, cached-file reuse and download cancellation. No new weights were downloaded.

Actual offline Turbo txt2img and Edit-Turbo img2img both produced 1024x1024 images on the RTX 5090. Editing used one reference and four steps; see simple-img2img-report.json. The installed older Edit-Turbo checkpoint was tested; no hotfix was downloaded.

The quality comparison used the same Turbo checkpoint, dimensions and seed with two different prompts; see quality-report.json and quality-*.png. This is a prompt comparison, not proof of a universal model-quality improvement. No automatic sharpening or silent prompt rewrite was added.


Optional 2px grid filter: 53 full-suite tests passed; the additional opt-in/stale-controls regression passed in the 7-test hook suite. Independent CPU implementation of the supplied shader's equation, verified against its direct 2D formula and synthetic alternating stripes/checkerboards. Border sampling explicitly clamps to edge. No claim of improved quality on every image. Source reviewed 2026-09-21; Reddit page could not be fetched, but the linked Pastebin workflow was inspected directly.


## LoRA and LoKr

72 tests passed with Forge Python and PEFT 0.18.0. Tests include the actual official Boogu LoRA loader with real PEFT and a small synthetic model, LoKr numerical equivalence (full/factorized), strength handling, packed-base preservation, Accelerate offloading, trunk locks and invalid layouts. No trained Boogu adapter was available locally, so full-size trained-adapter image quality remains unverified.

PEFT 0.18.0 was installed without dependency upgrades. Its optional TorchAO probe is bypassed only for this loader's own ordinary Linear targets during injection, then restored. Torch/CUDA and other engines were not changed.

