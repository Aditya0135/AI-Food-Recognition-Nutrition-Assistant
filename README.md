# 🍽️ AI Food Recognition & Nutrition Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end deep learning pipeline that **identifies food from photos** and instantly returns **nutrition facts** and **recipes**. Built on [ConvNeXt-Tiny](https://pytorch.org/vision/stable/models/convnext.html) fine-tuned on the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset (101 categories, ~101k images), served via an interactive **Streamlit** web app.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **Food recognition** | ConvNeXt-Tiny fine-tuned on Food-101 — 101 classes |
| 📊 **Top-5 predictions** | Confidence scores for the 5 most likely foods |
| 🥗 **Nutrition facts** | Calories, protein, carbs, fat, fiber & sugar via Spoonacular API |
| 🍳 **Recipe details** | Ingredients, step-by-step instructions & source link |
| 📸 **Two input modes** | Upload an image **or** use your webcam |
| ⚡ **GPU / CPU** | Runs on CUDA GPU or CPU transparently |
| 🐳 **Docker ready** | Single-container deployment |

---

## 🗂️ Project Structure

```
AI-Food-Recognition-Nutrition-Assistant/
├── src/AI_Food_Recognition_Nutrition_Assistant/
│   ├── components/          # Core logic
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── prepare_base_model.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── pipeline/            # Stage runners
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_preprocessing.py
│   │   ├── stage_03_prepare_base_model.py
│   │   ├── stage_04_train.py
│   │   └── stage_05_evaluate_model.py
│   ├── config/              # ConfigurationManager
│   ├── entity/              # Typed dataclass configs
│   └── utils/               # Helpers, API clients, Streamlit utils
├── config/config.yaml       # Path configuration
├── params.yaml              # Training hyperparameters & augmentation
├── main.py                  # Full training pipeline entry point
├── evaluate_only.py         # Evaluate a saved checkpoint only
├── app_streamlit.py         # Streamlit web app
├── requirements.txt         # Training / backend dependencies
├── requirements-streamlit.txt # Streamlit app dependencies
└── Dockerfile               # Container definition
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Aditya0135/AI-Food-Recognition-Nutrition-Assistant.git
cd AI-Food-Recognition-Nutrition-Assistant

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# For the Streamlit app only:
pip install -r requirements-streamlit.txt
```

### 2. Configure the Spoonacular API key

Sign up for a free key at <https://spoonacular.com/food-api>, then create `api_config.py` in the project root:

```python
# api_config.py  ← already in .gitignore, never commit real keys
SPOONACULAR_API_KEY = "your_key_here"
```

Alternatively, set it as a [Streamlit secret](https://docs.streamlit.io/library/advanced-features/secrets-management):

```toml
# .streamlit/secrets.toml
SPOONACULAR_API_KEY = "your_key_here"
```

---

## 🏋️ Training the Model

### Step 1 — Download the dataset

```bash
# Requires a Kaggle API token (kaggle.json) in ~/.kaggle/
python -c "
from AI_Food_Recognition_Nutrition_Assistant.pipeline.stage_01_data_ingestion import DataIngestionPipeline
DataIngestionPipeline().main()
"
```

The Food-101 dataset (~5 GB) is downloaded and extracted to `artifacts/data_ingestion/`.

### Step 2 — Run the full pipeline

```bash
python main.py
```

This executes five sequential stages:

| # | Stage | Output |
|---|---|---|
| 1 | Data Ingestion | Raw zip → extracted images |
| 2 | Data Preprocessing | Train/val/test splits + dataloaders |
| 3 | Prepare Base Model | ConvNeXt-Tiny with custom 101-class head |
| 4 | Train | Checkpoints in `artifacts/training/checkpoints/` |
| 5 | Evaluate | Metrics JSON, classification report, confusion matrix |

### Key hyperparameters (`params.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 60 | Maximum training epochs |
| `batch_size` | 32 | Batch size |
| `lr_backbone` | 3e-5 | Learning rate for ConvNeXt feature extractor |
| `lr_head` | 5e-4 | Learning rate for classification head |
| `patience` | 8 | Early-stopping patience |
| `use_mixup` | true | MixUp augmentation |
| `use_cutmix` | true | CutMix augmentation |
| `use_ema` | true | Exponential Moving Average |
| `label_smoothing` | 0.1 | Cross-entropy label smoothing |

---

## 🎯 Evaluate a Saved Checkpoint

If you already have a trained model and just want to run evaluation:

```bash
python evaluate_only.py
```

Results are saved to `artifacts/evaluation/`:
- `metrics.json` — top-1, top-5 accuracy, test loss
- `classification_report.txt` — per-class precision / recall / F1
- `confusion_matrix.npy` — full confusion matrix

---

## 🌐 Running the Web App

### Local

```bash
streamlit run app_streamlit.py
```

Open <http://localhost:8501> in your browser.

### Docker

```bash
# Build
docker build -t food-ai .

# Run (mount your trained model)
docker run -p 8080:8080 food-ai
```

> **Note:** The Docker image currently runs `app.py`. To serve the Streamlit app, update the `CMD` in `Dockerfile` to:
> ```dockerfile
> CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
> ```

---

## 🖥️ App Usage

1. **Upload** a food photo or **capture** one with your webcam.
2. Click **🔮 Analyze Food**.
3. The model returns the top-5 predicted food categories with confidence percentages.
4. If the top prediction exceeds the **70% confidence threshold**, nutrition facts and a recipe are fetched automatically.
5. Prediction history is stored in the sidebar for the current session.

---

## 🏗️ Architecture & Training Details

- **Backbone**: `convnext_tiny` (ImageNet-1K pretrained)
- **Head**: Single `nn.Linear(768 → 101)` replacing the default classifier
- **Optimiser**: AdamW with differential learning rates (backbone vs. head)
- **Scheduler**: OneCycleLR
- **Mixed precision**: `torch.amp.autocast` + `GradScaler` (auto-selects CUDA or CPU)
- **Augmentation**: RandomResizedCrop, RandAugment, ColorJitter, RandomErasing, MixUp, CutMix
- **Regularisation**: Label smoothing, weight decay, EMA weights

---

## 📦 Dependencies

| Purpose | Packages |
|---|---|
| Deep learning | `torch`, `torchvision` |
| Data / ML | `scikit-learn`, `numpy`, `scipy` |
| App | `streamlit`, `Pillow`, `opencv-python` |
| Config | `python-box`, `pyyaml`, `ensure` |
| Nutrition API | `requests` |
| Experiment tracking | `tensorboard`, `dvc` |

---

## 🔧 Development Workflow

```
1. Update config/config.yaml      ← artifact paths
2. Update params.yaml             ← hyperparameters
3. Update entity/config_entity.py ← typed config dataclasses
4. Update config/configuration.py ← ConfigurationManager
5. Implement / update components/ ← core logic
6. Implement / update pipeline/   ← stage runners
7. Run main.py                    ← end-to-end training
8. Deploy with app_streamlit.py   ← Streamlit UI
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — ETH Zürich
- [ConvNeXt](https://arxiv.org/abs/2201.03545) — Facebook AI Research
- [Spoonacular Food API](https://spoonacular.com/food-api) — recipe & nutrition data
