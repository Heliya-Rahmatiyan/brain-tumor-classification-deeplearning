# Deep Learning-Based Brain Tumor Classification from MRI Scans

An end-to-end, structured Deep Learning repository utilizing **ResNet18** to classify brain tumors from MRI scans into four distinct categories. This project transitions from frozen baseline evaluations to an optimized multi-model **Softmax Voting Ensemble**, backed by **Grad-CAM** visual explainability conditioned on clinical logic gates.

---

## Dataset Overview & Clinical Context

This project utilizes the **Sartaj Brain Tumor MRI Dataset**, a widely used open-source benchmark for brain tumor classification tasks.

### Dataset Characteristics
* **Imaging Modality**: Magnetic Resonance Imaging (MRI)
* **Image Type**: 2D axial brain MRI slices
* **Total Images**: 3,264
* **Classification Task**: Multi-class tumor classification (4 classes)

### Class Distribution
| Clinical Class | Number of Images |
| :--- | :---: |
| **Glioma Tumor** | 926 |
| **Meningioma Tumor** | 937 |
| **Pituitary Tumor** | 901 |
| **No Tumor** | 500 |
| **Total** | **3,264** |

### Clinical Categories
* **Glioma Tumor**: Tumors originating from glial cells within the central nervous system.
* **Meningioma Tumor**: Tumors arising from the meninges surrounding the brain and spinal cord.
* **Pituitary Tumor**: Tumors located in the pituitary gland at the base of the brain.
* **No Tumor**: Healthy control MRI scans used for baseline comparison.

### Dataset Challenges
The dataset presents several real-world machine learning challenges:
1. Moderate class imbalance, particularly in the `no_tumor` category.
2. Significant visual similarity between certain tumor subtypes.
3. Limited dataset size compared to large-scale natural image benchmarks.
4. Domain shift between ImageNet pretraining and medical imaging data.

*These challenges motivated the extensive transfer learning, fine-tuning, ensemble learning, and explainability experiments documented throughout this repository.*

---

## Project Architecture (Step-by-Step Pipeline)

The repository is organized into 10 sequential, production-grade notebooks tracking the entire R&D lifecycle:

### Phase 1: Data Engineering & Baseline Setup
* **`01_data_extraction.ipynb` (Step 1)**: Automated data ingestion, integrity checks, and train/test splitting.
* **`02_data_preprocessing.ipynb` (Step 2)**: Standardization pipeline including image resizing, normalization, and balanced medical data augmentations.
* **`03_resnet_baseline_pipeline.ipynb` (Step 3)**: PyTorch workflow design setting up the data loaders, loss functions, and core training loops.
* **`04_transfer_learning.ipynb` (Step 4)**: Initializing the ResNet18 architecture with ImageNet weights and reshaping the final Fully Connected (FC) layer for 4 classes.

### Phase 2: Iterative Optimization & Feature Tuning
* **`05_resnet_baseline_evaluation.ipynb` (Step 5 - V1)**: *Native Balance Test*. Model backbone is completely frozen; only the new classifier layer is trained for 4 epochs to evaluate raw ImageNet feature response on medical data ($Accuracy \approx 56\%$).
* **`05_resnet_fine_tuning_layer4.ipynb` (Step 5 - V2)**: *Partial Fine-Tuning Phase*. Layers 1-3 remain frozen while Layer 4 and the FC layer are opened. This setup triggered a heavy shortcut learning loop, leading to severe overfitting on training noise ($Train\ Acc \approx 98\%$, but missing crucial Glioma features on evaluation).
* **`06_resnet_full_finetuning.ipynb` (Step 6)**: *Full Fine-Tuning Phase*. The entire architecture is unfrozen to break the Step 5 overfitting anchor. Class weights (2.5x penalty for Glioma) are injected into `CrossEntropyLoss` to tackle high-mismatch boundaries, and model saving is switched to track `Validation Accuracy` ($Accuracy \rightarrow 84\%$).
* **`07_resnet_layered_lr.ipynb` (Step 7)**: *Discriminative Learning Rates Exploration*. Early layers were strictly bound ($lr=1e-6$) while deeper blocks were heavily updated ($lr=1e-4$). Acting as a "feature anchor," this created a highly specific voter for the final ensemble ($Accuracy \approx 80.4\%$).

### Phase 3: Ensemble Deployment & Clinical Explainability
* **`08_model_ensemble_inference.ipynb` (Step 8)**: *Softmax Voting System*. Merges predictions from Step 6 and Step 7 via probability averaging, boosting overall metrics to a peak performance of **85.02% Accuracy**.
* **`09_model_interpretability_gradcam.ipynb` (Step 9)**: Explains model decisions using Grad-CAM heatmaps generated from the Step 6 backbone, guarded by clinical logic.

---

## Performance Breakthrough & Validation Analysis

### The Power of Collective Intelligence (Ensemble)
While the Step 6 model achieved an impressive $83.5\%$ single-model accuracy, integrating it with the Step 7 specialized model via a **Softmax Ensemble** pushed the final system to **85.02% Accuracy** ($Macro\ F1 = 0.84$). 

| Target Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma Tumor** | **1.00** | 0.48 | 0.65 | 100 |
| **Meningioma Tumor** | 0.76 | **0.99** | 0.86 | 115 |
| **No Tumor (Healthy)** | 0.82 | **1.00** | 0.90 | 105 |
| **Pituitary Tumor** | 0.99 | 0.91 | 0.94 | 74 |
| *System Accuracy* | | | **0.85** | **394** |

### Clinical Significance of the Metrics
1. **Zero False Positives for Glioma (Precision 1.00)**: If the system diagnoses a Glioma, the confidence level is absolute. This prevents highly toxic, unnecessary immediate interventions.
2. **Flawless Healthy Scan Isolation (Recall 1.00)**: The ensemble achieved a perfect score on `no_tumor`, ensuring no healthy patient is mistakenly exposed to oncological panic or false treatment paths.

---

## Failed Hypotheses & Engineering Insights (Ablation Studies)

True machine learning engineering is defined by testing boundaries. This project documents critical architectural bottlenecks and failures that led to the final optimal system:

### 1. The Step 5 Partial Fine-Tuning Bottleneck
In `05_resnet_fine_tuning_layer4.ipynb`, restricting the network's optimization capacity by only unfreezing `layer4` on a small medical dataset forced the model to memorize high-frequency noise and shortcuts rather than learning true clinical pathology. This structural failure proved that full network adaptation is mandatory for complex domain shifts like MRI scans.

### 2. The EfficientNet-B0 & RandomErasing Failure
We attempted to upgrade the backbone to `EfficientNet-B0` while utilizing `RandomErasing` to force robust texture mapping. This caused a catastrophic validation drop ($Val\ Acc \rightarrow 74\%$):
* **Spatial Destructiveness**: Unlike general object recognition (cats/dogs), tumor identification relies on subtle pixel boundaries and low-contrast variations. `RandomErasing` literally wiped out crucial pathology microstructures, blinding the network.
* **Data Scarcity vs. Model Depth**: EfficientNet's internal attention mechanisms (Squeeze-and-Excitation blocks) require massive datasets to converge. On our refined dataset, it immediately memorized noise, causing a severe overfitting loop.

### 3. The Rigid Anchor Effect of Layered Learning Rates
In Step 7, setting early layers to a frozen-like pace ($lr=1e-6$) while unfreezing the classifier layer caused the model to lock down. ImageNet structural features acted as an unyielding anchor, preventing the network from adapting its early filters to the specific contrast gradients of MRI imaging. However, this yielded an exceptionally conservative predictor with high class-precision, which we successfully recycled as a diversity booster in the **Step 8 Ensemble**.

---

## Explainable AI (XAI): Clinical-Grade Grad-CAM

To cross the trust barrier in clinical software, **Step 9** deploys Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the network's spatial attention.

**Clinical Logic Routing:**
* **Tumor Predicted**: The pipeline overlays dynamic heatmaps precisely targeting the detected pathology zones to guide the radiologist and ensure clinical accountability.
* **No Tumor Predicted**: The system triggers a clinical gate that completely suppresses the visualization layer, rendering a clean, distraction-free scan to prevent cognitive overload during negative findings (both True Negatives and False Negatives).

---

## ⚠️ Limitations & Clinical Disclaimer

While the proposed pipeline and ensemble system achieved strong performance on the evaluation set, several critical boundaries must be acknowledged:

1. **Single Cohort Evaluation**: Evaluation was strictly performed on a single public dataset; no external validation cohorts from different clinical centers or MRI scanners were available to test cross-site generalization.
2. **2D Dimensionality**: The MRI scans were processed as independent 2D axial slices rather than continuous 3D volumetric studies, which limits the spatial contextual awareness of the network.
3. **Data Scale**: The dataset volume remains relatively small compared to massive natural image benchmarks, leaving room for further optimization via larger multi-institutional data.
4. **No Clinical Readiness Implied**: Performance metrics and visual heatmaps should not be interpreted as evidence of true clinical validation or readiness. 
