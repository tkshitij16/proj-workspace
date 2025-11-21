Here’s a polished `README.md` you can drop straight into the top-level `proj` repo:

````markdown
# GAN TinyML FPGA Workspace

This repository is a **lightweight workspace wrapper** for a full pipeline that:

- Starts from the **MIT GAN Compression** project (teacher + compressed pix2pix)
- Trains a **tiny student network** (UNet-style) to mimic the compressed GAN
- Exports the student to **ONNX** for **TinyML / FPGA deployment**
- Compares latency and quality across **CPU (WSL)**, **GPU server**, and **FPGA board**

> ⚠️ This workspace does **not** contain the full training code itself.  
> It expects the following two repositories to live as sibling folders:

- `gan-compression/` – upstream MIT repo (teacher + compressed GAN)
- `gan-tinyml-fpga/` – your student + ONNX + UI + FPGA integration code

---

## 1. Repository Layout

Expected directory structure under `~/proj`:

```text
~/proj/
  README.md              # This file

  gan-compression/       # Upstream MIT repo (teacher & compressed GAN)
                         # - Full & compressed pix2pix models
                         # - FID / latency scripts

  gan-tinyml-fpga/       # Your main repo (student + deployment)
                         # - Student UNet training
                         # - ONNX export
                         # - Gradio UI
                         # - (Optional) FPGA integration scripts
````

This repo (`proj-workspace`) is just the **meta README** describing how the pieces fit together and how to reproduce experiments on different machines.

---

## 2. Getting the Code on a New Machine

On any new machine (WSL laptop, GPU server, or FPGA build machine), do:

```bash
mkdir -p ~/proj
cd ~/proj

# 1) Clone the MIT teacher/compressed GAN repo
git clone https://github.com/mit-han-lab/gan-compression.git

# 2) Clone your student + TinyML / FPGA repo
git clone https://github.com/tkshitij16/gan-tinyml-fpga.git
```

Now the machine has the same layout as the original development environment.

> You **don’t need** to nest everything into a single giant repo. Treat this folder as a **workspace** that ties the two projects together.

---

## 3. Roles of Each Sub-Repository

### `gan-compression/` (teacher + compressed GAN)

* Original MIT codebase for **GAN compression** (e.g., pix2pix on edges2shoes).
* Used to:

  * Run **full** and **compressed** GAN inference.
  * Compute **FID** and other quality metrics.
  * Measure **latency** on CPU and GPU.

Typical scripts (examples):

```bash
# Inference
bash scripts/pix2pix/edges2shoes-r/test_full.sh
bash scripts/pix2pix/edges2shoes-r/test_compressed.sh

# Latency
bash scripts/pix2pix/edges2shoes-r/latency_full.sh
bash scripts/pix2pix/edges2shoes-r/latency_compressed.sh
```

---

### `gan-tinyml-fpga/` (student + ONNX + UI + FPGA)

* Your core project repo:

  * Loads teacher/compressed outputs as targets.
  * Trains a **small StudentUNet** (TinyML-friendly).
  * Exports the final model to **ONNX**.
  * Provides a **Gradio UI** for quick visual comparison.
  * Contains or interfaces with scripts for **FPGA deployment**.

Typical flow (examples):

```bash
cd ~/proj/gan-tinyml-fpga

# Train student on compressed GAN outputs
python train_student.py \
  --data-root data/edges2shoes_student \
  --epochs 30 \
  --base-ch 16

# Export ONNX
python compile/export_onnx.py

# Evaluate student vs teacher
python eval/eval_student_metrics.py

# Run Gradio UI
python ui/gradio_app.py
```

---

## 4. Multi-System Workflow (WSL ⇄ GPU ⇄ FPGA)

Think of **GitHub** as the **single source of truth**, and each machine as a role:

### A. WSL Laptop (CPU Dev Box)

**Main purpose:** development, debugging, and initial experiments.

You typically do:

* Clone both repos and set up Python envs.
* Generate **student training dataset**:

  * Edge inputs + compressed GAN outputs as pseudo-labels.
* Train a **small StudentUNet** for sanity-check epochs.
* Export **ONNX**.
* Build and test the **Gradio UI**.
* Measure **CPU latency** for the student model.

Once the code + configs are stable:

* Commit to `gan-tinyml-fpga`.
* Push to GitHub so GPU and FPGA machines can pull the same state.

---

### B. GPU Server

**Main purpose:** heavy training + teacher/compressed baselines.

On the GPU machine:

```bash
cd ~/proj
git clone https://github.com/mit-han-lab/gan-compression.git
git clone https://github.com/tkshitij16/gan-tinyml-fpga.git
```

Set up a GPU-enabled environment (PyTorch + CUDA) and:

1. **Re-run heavy teacher/compressed experiments:**

   * Full + compressed GAN inference (`test_full.sh`, `test_compressed.sh`)
   * Full + compressed GAN **latency** on GPU (`latency_full.sh`, `latency_compressed.sh`)

2. **Train the “final” student model on GPU:**

   * More epochs, maybe larger `base_ch` (e.g., 32).
   * Re-export ONNX and recompute metrics.

3. **Commit and push:**

   * Only code, configs, and small artifacts.
   * Use Git LFS or external storage if you keep large checkpoints.

> You **do not** have to regenerate the student dataset from scratch if you can copy the prepared dataset from WSL to the GPU machine (`scp`, shared drive, etc.).

---

### C. FPGA Board

**Main purpose:** deployment, latency, and (optionally) power measurements.

The FPGA environment **only needs**:

* Final student model in a deployable format:

  * `student_unet_simplified.onnx`
  * Quantized variant if required by the toolchain.
* A small C/C++/Python **host program** that:

  * Loads test edge images.
  * Feeds them to the FPGA accelerator.
  * Reads back outputs and measures latency/energy.

It does **not** need:

* The full `gan-compression` training pipeline.
* FID computations.
* Any GAN training code.

FPGA is strictly for **inference**.

---

## 5. What to Run Where (Cheat Sheet)

### By System

| Task                                        | WSL Laptop (CPU) | GPU Server | FPGA Board |
| ------------------------------------------- | :--------------: | :--------: | :--------: |
| Clone repos & basic setup                   |         ✅        |      ✅     |      ✅     |
| Generate student dataset                    |         ✅        |  optional  |      ❌     |
| Student training (small-scale / debug)      |         ✅        |      ✅     |      ❌     |
| Full + compressed GAN inference             |     optional     |      ✅     |      ❌     |
| Full + compressed GAN latency               |     optional     |      ✅     |      ❌     |
| Final student training (long, tuned)        |       maybe      |      ✅     |      ❌     |
| Export ONNX                                 |         ✅        |      ✅     |      ❌     |
| Gradio UI                                   |         ✅        |  optional  |      ❌     |
| FPGA compilation (ONNX → bitstream/engine)  |         ❌        |   depends  |      ✅     |
| FPGA inference + latency/energy measurement |         ❌        |      ❌     |      ✅     |

---

## 6. How to Present Results in a Report

You can structure your report around **hardware setups** and **comparative tables**.

### Hardware Setups

* **CPU (WSL Laptop)**

  * CPU model, RAM, OS
  * Framework: PyTorch + ONNX Runtime
* **GPU Server**

  * GPU model(s), CUDA version, driver
  * Framework: PyTorch, mixed precision settings
* **FPGA Board**

  * Board name, FPGA fabric, clock frequency
  * Toolchain (e.g., Vitis AI, TVM, hls4ml, vendor SDK)

### Example Latency Table

| Model              |  Hardware | Latency (ms) | Speedup vs Full | Notes                        |
| ------------------ | --------: | -----------: | --------------: | ---------------------------- |
| Full pix2pix GAN   |       GPU |   T_full_gpu |            1.0× | Highest image quality        |
| Compressed GAN     |       GPU |   T_comp_gpu |  ~9× fewer MACs | Slight FID drop              |
| StudentUNet (ours) | CPU (WSL) |         26.7 |               – | Tiny, no GPU required        |
| StudentUNet (ours) |      FPGA |       T_fpga |       ?× vs CPU | Low-power, embedded-friendly |

### Example Quality Table

| Model              | FID vs Real | PSNR vs Compressed | Comment                          |
| ------------------ | ----------: | -----------------: | -------------------------------- |
| Full pix2pix GAN   |      F_full |                  – | Best visual quality              |
| Compressed GAN     |      F_comp |                  – | Slight quality loss, large speed |
| StudentUNet (ours) |           – |            15.6 dB | Approximates teacher; tunable    |

---

## 7. Summary

This workspace repo is intentionally minimal:

* It **documents** how `gan-compression` and `gan-tinyml-fpga` work together.
* It defines a clear **multi-system workflow**:

  * Develop on **WSL/CPU**
  * Benchmark and train at scale on **GPU**
  * Deploy and measure on **FPGA**
* It gives you a reproducible way to:

  > clone → train → export → deploy → compare

If you add new scripts, configs, or diagrams that describe the end-to-end pipeline, this `README.md` is the place to link them.
