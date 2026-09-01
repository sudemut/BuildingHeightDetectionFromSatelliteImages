# 🏙️ Building Height Estimation from Satellite Imagery

Deep learning-based building height estimation from **RGB satellite images**, developed as a Computer Science Senior Project at Özyeğin University.

The project explores how accurately building heights can be inferred from a **single satellite image without LiDAR or stereo imagery**. We implemented and compared **five different deep learning architectures**, experimented with multiple loss functions and normalization strategies, and built a desktop GUI for model inference, visualization, and comparison.

Our best-performing model achieved:

**MAE: 3.54 m · RMSE: 5.52 m · R²: 0.72**

---

##  Project Overview

Obtaining accurate 3D information about cities is important for applications such as:

* Urban planning
* Disaster response
* Telecommunications
* Environmental modeling
* 3D city reconstruction

Traditional methods such as **LiDAR** provide highly accurate measurements, but they are expensive and geographically limited.

This project investigates a different approach:

> **Can a neural network estimate building height using only RGB satellite imagery?**

Given a satellite image, the system predicts a **dense height map**, where every building pixel represents the estimated height above ground level in meters.

---

##  Models

We implemented and evaluated five different architectures.

| Model         | Architecture                      | Key Idea                                         |        MAE |       RMSE |       R² |
| ------------- | --------------------------------- | ------------------------------------------------ | ---------: | ---------: | -------: |
| **Model 1**   | DeepLabV3+ / ResNet50             | Background Regularization + Scale-Invariant Loss |     5.00 m |     8.56 m | **0.77** |
| **Model 2**   | Dual-Head U-Net / HRNet-W18       | Gradient Preservation + Segmentation             |     4.30 m |     7.50 m |     0.71 |
| **Model 3 ⭐** | Dual-Head U-Net / EfficientNet-B4 | BerHu Loss + Percentile Normalization            | **3.54 m** | **5.52 m** |     0.72 |
| **Model 4**   | FPN / ResNet50                    | Multi-Task Height + Edge Detection               |     4.34 m |     7.68 m |     0.64 |
| **Model 5**   | SegFormer / MIT-B4                | Ordinal Regression + Uncertainty Estimation      |     5.20 m |     8.10 m |     0.65 |

###  Best Model

The **EfficientNet-B4 Dual-Head model** achieved the lowest prediction error.

Its performance comes from combining:

* EfficientNet-B4 feature extraction
* Explicit building segmentation
* BerHu regression loss
* Background regularization
* Percentile-based height normalization

The final prediction combines height estimation with the predicted building probability, helping suppress false height predictions on roads, vegetation, and other non-building areas.

---

##  Dataset

The models were trained using the **IEEE GRSS Data Fusion Contest 2023 - Track 2 dataset**.

The dataset contains:

* RGB satellite imagery
* DSM-derived height maps
* Building-level spatial information

Training-set height statistics:

| Statistic          |    Value |
| ------------------ | -------: |
| Minimum Height     |    0.0 m |
| Maximum Height     | 183.17 m |
| Mean Height        |  18.04 m |
| Standard Deviation |  12.16 m |

---

##  Data Augmentation

To improve generalization, the training pipeline uses **Albumentations** with:

* Horizontal flips
* Vertical flips
* Random 90° rotations
* Color jitter
* Gaussian noise
* Gaussian blur
* ImageNet normalization

---

##  Architecture Experiments

### Model 1 — DeepLabV3+

Uses a **ResNet50 encoder** and Atrous Spatial Pyramid Pooling for multi-scale feature extraction.

A custom background regularization term penalizes non-zero height predictions outside building regions.

```text
Loss = Building Loss + λ × Background Loss
```

This model produced particularly strong background suppression, with background predictions typically around **0.28–1.06 m**.

---

### Model 2 — Dual-Head U-Net + HRNet

Uses an **HRNet-W18 encoder** with separate prediction heads for:

1. Building height regression
2. Building segmentation

A gradient-preservation loss based on Sobel filters encourages sharper building boundaries.

```text
Loss =
L1
+ λgrad × Gradient Loss
+ λseg × Segmentation Loss
+ λbg × Background Loss
```

This model performed particularly well on taller buildings.

---

### Model 3 — EfficientNet-B4 + BerHu ⭐

The best-performing architecture.

A Dual-Head U-Net with an **EfficientNet-B4 encoder** jointly predicts:

* Building height
* Building probability

BerHu loss combines L1 and L2 behavior, making the model more robust to both small and large prediction errors.

```text
Loss =
BerHu
+ λsmooth × Smoothness Loss
+ λseg × Segmentation Loss
+ λbg × Background Loss
```

**Best MAE: 3.54 meters**

---

### Model 4 — FPN + Edge Detection

A multi-task **Feature Pyramid Network** predicts both:

* Building heights
* Building boundaries

The edge detection branch uses Focal BCE and morphological edge extraction.

Performance:

```text
Edge F1        0.81
Edge IoU       0.68
Precision      0.80
Recall         0.88
```

Explicit edge supervision produces sharper building boundaries, although the additional task slightly increases height prediction error.

---

### Model 5 — SegFormer + Ordinal Regression

Instead of predicting height directly, this model converts the regression problem into an **ordered classification problem**.

The height range is divided into **100 ordinal bins**.

```text
P(height > threshold_1)
P(height > threshold_2)
...
P(height > threshold_100)
```

The final height is reconstructed from these probabilities.

An additional benefit of this approach is the ability to derive **uncertainty maps using prediction entropy**.

Ordinal classification accuracy reached approximately **97%**, although convergence was slower than direct regression approaches.

---

#  Model Comparison GUI

Alongside the training pipeline, we developed a custom GUI for running and comparing trained models.

The application automatically identifies model architectures from their checkpoints and applies the appropriate preprocessing and denormalization strategy.

### Prediction

The prediction interface supports:

* Loading multiple trained models
* Selecting satellite images
* Running all models simultaneously
* Height-map visualization
* Per-model prediction statistics
* Height histograms

### Analytics

Provides visual comparisons including:

* Height distributions
* Box plots
* MAE / RMSE comparison
* Prediction mean, maximum and standard deviation

### Comparison

Allows predictions to be inspected side-by-side using:

* Contour overlays
* Difference maps
* Error visualization
* Original satellite imagery

### Model Information

Displays each model's:

* Architecture
* Encoder
* Loss function
* Normalization strategy
* Training epoch
* Validation metrics
* Height statistics

---

##  Exporting Predictions

Prediction results can be exported in multiple formats:

```text
.npy   → Raw NumPy height arrays
.png   → Colorized height maps
.json  → Model configuration and prediction statistics
```

---

##  Training Configuration

The experiments use a shared training setup:

| Parameter      | Value             |
| -------------- | ----------------- |
| Optimizer      | AdamW             |
| Learning Rate  | 1e-4              |
| Weight Decay   | 1e-4              |
| Scheduler      | Cosine Annealing  |
| Batch Size     | 8–16              |
| Max Epochs     | 50                |
| Early Stopping | 10–15 epochs      |
| Acceleration   | Apple Silicon MPS |

Training was performed on **Apple Silicon using PyTorch Metal Performance Shaders (MPS)**.

---

##  Technologies

The project primarily uses:

* **Python**
* **PyTorch**
* **Segmentation Models PyTorch**
* **Albumentations**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **EfficientNet**
* **DeepLabV3+**
* **HRNet**
* **Feature Pyramid Networks**
* **SegFormer**
* **Apple Metal Performance Shaders**

---

##  Key Findings

### EfficientNet-B4 performed best

The EfficientNet-based model achieved the lowest overall error:

```text
MAE   = 3.54 m
RMSE  = 5.52 m
R²    = 0.72
```

### Background regularization matters

Without explicit background handling, models produced approximately **3–5 meters of height leakage** onto roads, trees, and other non-building regions.

Background-aware models substantially reduced this effect.

### Edge supervision improves geometry

Model 4 demonstrated that explicitly learning building boundaries creates sharper predictions.

However, multi-task optimization introduced a small trade-off in height accuracy.

### Tall buildings remain difficult

Buildings above approximately **40 meters** are frequently underestimated because high-rise buildings are underrepresented in the training distribution.

The models perform best in the **10–30 meter range**.

---

##  From Segmentation to Height Estimation

This work builds on our previous building segmentation project.

|                   | Previous Stage        | Current Project            |
| ----------------- | --------------------- | -------------------------- |
| Task              | Building Segmentation | Building Height Estimation |
| Problem Type      | Classification        | Regression                 |
| Output            | Binary Building Mask  | Continuous Height Map      |
| Main Architecture | UNet++                | 5 Compared Architectures   |
| Best Result       | IoU ≈ 0.75            | MAE ≈ **3.54 m**           |

The first stage focused on answering:

> **Where are the buildings?**

This project extends that problem to:

> **Where are the buildings, and how tall are they?**

---

##  Future Work

Potential improvements include:

* Combining EfficientNet-B4 with edge-aware supervision
* Training ordinal models for more epochs
* Increasing representation of high-rise buildings
* Testing larger Transformer-based encoders
* Improving boundary-aware regression
* Adding confidence-aware prediction filtering
* Extending the pipeline toward full 3D city reconstruction

---

##  Authors

**Sude Mut**
**Umut Ardıl Murat**

Department of Computer Science
Faculty of Engineering
Özyeğin University

**Supervisor:** Hasan Fehmi Ateş

---

##  References

The project builds on research including:

* U-Net
* UNet++
* DeepLabV3+
* EfficientNet
* Feature Pyramid Networks
* SegFormer
* Scale-Invariant Depth Estimation
* Deep Ordinal Regression

Dataset:

**IEEE GRSS Data Fusion Contest 2023 — Track 2**

---

##  Summary

This project explores several approaches to extracting **3D building information from ordinary RGB satellite imagery**.

Rather than implementing a single model, we designed the project as an experimental framework to compare architectures, losses, normalization strategies, segmentation-assisted regression, edge-aware learning, and ordinal prediction.

The result is an end-to-end system covering:

**Satellite Image → Deep Learning Model → Building Height Map → Analysis & Visualization**

with a best observed **Mean Absolute Error of 3.54 meters**.
