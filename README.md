# Soil Multitask Distillation

**Unifying Chemometric for Robust Soil Spectroscopy via Multi‑Task Distillation (Vis‑NIR + XRF + LIBS)**

A compact, unified student model distilled from strong, per‑attribute chemometric teachers to jointly predict nine soil properties:

> **Clay, OM, CEC, pH, V, exP, exK, exCa, exMg**

This codebase reproduces teacher pipelines inspired by Tavares *et al.* and trains a multi‑task, multi‑sensor student network that consolidates prediction across three modalities (Vis‑NIR, XRF, LIBS). The CLI lets you train, evaluate, and visualize results with a single YAML config.

---

## Quickstart

```bash
# 1) Install
pip install -r requirements.txt

# 2) Train student with knowledge distillation
python main.py train --config configs/default.yaml

# 3) Evaluate a saved checkpoint on the validation split
python main.py evaluate --config configs/default.yaml \
  --checkpoint artifacts/student_seed34.pt

# 4) Make per‑task scatter plots
python main.py visualize --config configs/default.yaml \
  --checkpoint artifacts/student_seed34.pt --output figures
```

**Outputs:** checkpoints are saved to `artifacts/student_seed{SEED}.pt`. Visualizations are written to the directory provided via `--output` (default: `figures/`).

> **Hardware:** Runs on CPU or GPU (automatically uses CUDA if available).

---

## Repository Layout

* `configs/` — experiment settings (see `default.yaml`)
* `data/` — loaders and preprocessing
* `models/` — teacher pipelines, student architecture, fusion blocks
* `distillation/` — losses and trainer
* `evaluation/` — metrics and plotting utilities
* `utils/` — logging and seed helpers
* `data_unzip/` — expected location of the dataset files (see below)
* `artifacts/` — checkpoints written here during training
* `main.py` — CLI entry point

---

## Dataset Sources & Access

This repository builds on the publicly available tropical soils benchmark released by Tavares and colleagues.

* **dataset**: *Spectral data of tropical soils using dry‑chemistry techniques (VNIR, XRF, and LIBS): A dataset for soil fertility prediction*, Data in Brief 41:108004 (2022). DOI: 10.1016/j.dib.2022.108004.

### License & attribution

* Respect the license distributed with the dataset and **cite both works above** in any derivative publication.

### Where the files go

By default (`configs/default.yaml`), the code searches `data_unzip/` for the dataset files (names are matched case‑insensitively):

* `soil fertility data.txt`
* `VNIR data.txt`
* `XRF data.txt`
* `LIBS data.txt`

> If you keep the default paths and filenames, **no extra steps** are required; the loader will discover these files automatically.

---

## Configuration File — **YAML Schema**

All experiments are driven by a single YAML. The provided `configs/default.yaml` is a complete, runnable example.

### Top‑level keys

```yaml
seed: 34                 # default seed when `seeds` is not set
seeds: [34, 1234, 5219]  # run multiple seeds (optional)

logging:
  level: INFO            # logging level

artifacts:
  dir: artifacts         # where checkpoints are saved
```

### `data` block

Controls preprocessing, LIBS windowing, and split strategy.

```yaml
data:
  root: data_unzip       # folder to scan for the four .txt files
  unzip_dir: data_unzip  # kept for compatibility; scanning uses `root`

  # Savitzky–Golay for VNIR (applied after SNV + max‑norm)
  sg_window: 11          # window length
  sg_poly: 2             # polynomial order
  sg_deriv: 1            # derivative order

  # LIBS wavelength limits for the "full" student branch
  libs_band: [200.0, 540.0]

  # Task‑specific LIBS sub‑ranges used by teacher pipelines
  libs_ranges:
    Clay: [[220.84, 222.21], [246.77, 248.29], [251.37, 252.94]]
    OM:   [[295.08, 296.92]]
    CEC:  [[201.31, 272.37], [287.89, 298.73], [414.15, 416.63]]
    pH:   [[220.84, 222.21], [237.82, 239.27], [391.78, 394.22]]
    V:    [[201.31, 272.37], [279.18, 300.60], [396.59, 399.10]]
    exP:  [[214.14, 215.44]]
    exK:  [[375.28, 377.53]]
    exCa: [[389.37, 391.77]]
    exMg: [[284.36, 286.14]]

  # Split (Kennard–Stone on targets)
  train_ratio: 0.7
  split_seed: 34
```

### `teacher` block

Per‑task chemometric configuration. **`method`** must be one of:

* `VNIR-PLS` — PLS on VNIR only (`V_PLS`)
* `VNIR+XRF-SF-PLS` — concatenated VNIR+XRF with PLS (`VX_SF`)
* `LIBS-PLS` — PLS on LIBS only (`L_PLS`)
* `VNIR+LIBS-GR` — PLS(VNIR) + PLS(LIBS) + concatenated PLS(VNIR+LIBS) stacked via linear meta‑learner (`V_PLS`, `L_PLS`, `VL_SF`)
* `V+X+L-SF-PLS` — concatenated VNIR+XRF+LIBS with PLS (`VXL_SF`)

For each referenced sub‑block, provide a scaler and component count:

```yaml
teacher:
  Clay:
    method: V+X+L-SF-PLS
    VXL_SF: {scaler: std,  ncomp: 4}
  OM:
    method: V+X+L-SF-PLS
    VXL_SF: {scaler: std,  ncomp: 3}
  CEC:
    method: VNIR+LIBS-GR
    V_PLS:  {scaler: auto, ncomp: 9}
    L_PLS:  {scaler: auto, ncomp: 9}
    VL_SF:  {scaler: std,  ncomp: 5}
  pH:
    method: VNIR-PLS
    V_PLS:  {scaler: auto, ncomp: 7}
  V:
    method: VNIR+XRF-SF-PLS
    VX_SF:  {scaler: auto, ncomp: 7}
  exP:
    method: VNIR+LIBS-GR
    V_PLS:  {scaler: auto, ncomp: 9}
    L_PLS:  {scaler: auto, ncomp: 7}
    VL_SF:  {scaler: std,  ncomp: 5}
  exK:
    method: VNIR+XRF-SF-PLS
    VX_SF:  {scaler: auto, ncomp: 9}
  exCa:
    method: LIBS-PLS
    L_PLS:  {scaler: auto, ncomp: 7}
  exMg:
    method: LIBS-PLS
    L_PLS:  {scaler: auto, ncomp: 13}
```

**Scalers:** `auto` (autoscale to zero‑mean/unit‑variance) or `std` (standardization). Component counts are the PLS latent dimensions.

### `student` block

Defines the multi‑branch Inception‑style 1D‑CNN, fixed prior gates, and MoE fusion.

```yaml
student:
  branches:
    V:                 # VNIR branch
      size: m          # 's' (small) or 'm' (medium)
      separable: true  # depthwise separable conv in Inception blocks
      cap: 224         # channel cap for branch width
      norm: bn         # one of: bn | ln | gn
      gem_p: 3.5       # GeM pooling exponent
      gem_mix: 0.4     # GeM–TopK mixing weight (0..1)
      enable_peak: false
      peak_kset: [7, 11, 15]
    X:                 # XRF branch
      size: m
      separable: false
      cap: 192
      norm: bn
      gem_p: 3.5
      gem_mix: 0.6
      enable_peak: false
      peak_kset: [7, 11, 15]
    L:                 # LIBS branch (full band)
      size: s
      separable: false
      cap: 224
      norm: gn
      gem_p: 3.5
      gem_mix: 0.5
      enable_peak: false
      peak_kset: [7, 11, 15]

  moe:
    hidden: 128        # hidden size in each expert
    act: relu          # relu | gelu
    tau: 1.0           # softmax temperature for expert mixing
```

### `training` & `evaluation` blocks

```yaml
training:
  batch: 24            # batch size
  lr: 0.001            # AdamW learning rate
  wd: 0.0005           # weight decay
  epochs: 160          # max epochs
  patience: 25         # early stopping

  # MixUp schedule (alpha anneals after `mix_turn` of training)
  mix_alpha_start: 0.4
  mix_alpha_end: 0.15
  mix_turn: 0.6        # fraction of epochs with `alpha_start`

  # Loss weights for KD and GT regression (all on normalized targets)
  out_kd_w: 1.0        # student vs teacher (multi‑modal) outputs
  feat_kd_w: 0.10      # student vs single‑modality teacher features
  gt_w: 0.15           # student vs ground truth

  clip: 0.5            # grad‑norm clipping
  num_workers: 0       # DataLoader workers

evaluation:
  metrics: [R2, RMSE]  # aggregated per task + macro mean
```

---

## Reproducibility & Artifacts

* **Splits:** Kennard–Stone with ratio `data.train_ratio` and seed `data.split_seed`.
* **Multiple runs:** set `seeds: [...]` to train and save one checkpoint per seed.
* **Checkpoints:** contain model weights, split indices, target normalization stats, and VNIR/XRF scalers—sufficient to re‑evaluate with `main.py evaluate`.

---

## Troubleshooting

* **Dataset not found:** Ensure the four `.txt` files are present under the folder pointed to by `data.root` (default: `data_unzip/`). Filenames are matched case‑insensitively and do not need to be exact, as long as they contain the substrings `soil fertility`, `vnir`, `xrf`, and `libs`.
* **CUDA not used:** The code automatically selects CUDA if available; otherwise it runs on CPU. No extra flags are needed.

---

## Citation

If you use this code, please cite the dataset listed in **Dataset Sources & Access** and your derived work accordingly.
