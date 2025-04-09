# Rice Leaf Disease Classification Project

[![GitHub stars](https://img.shields.io/github/stars/Jack-622/Rice-Leaf-Disease-Classification?style=social)](https://github.com/Jack-622/Rice-Leaf-Disease-Classification/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Jack-622/Rice-Leaf-Disease-Classification?style=social)](https://github.com/Jack-622/Rice-Leaf-Disease-Classification/fork)
[![Issues](https://img.shields.io/github/issues/Jack-622/Rice-Leaf-Disease-Classification)](https://github.com/Jack-622/Rice-Leaf-Disease-Classification/issues)


Welcome to the **Rice Leaf Disease Classification** repository 🌾. This project provides a two-stage deep-learning-based pipeline to identify and classify diseases in rice leaves. By leveraging TensorFlow, MobileNet, and a user-friendly Gradio interface, it aims to simplify the diagnostic process for researchers, farmers, and anyone interested in plant disease detection.

---

## 📂 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset Preparation](#-dataset-preparation)  
3. [Code Attribution](#code-attribution)  
4. [Project Structure](#project-structure)  
5. [Features](#features)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [How It Works](#how-it-works)  
9. [Utilized Models Details](#utilized-model-details)  
10. [Notebooks Explanation](#notebooks-explanation)  
11. [Contributing](#contributing)  
12. [License](#license)  
13. [Contact](#contact)

---

## Project Overview

Diseases in rice crops can significantly impact yield and quality if not identified and treated promptly. This project addresses that challenge with a two-stage model:

1. **Stage 1 Model:** Determines if the input image contains a rice leaf or not.  
2. **Stage 2 Model:** If Stage 1 identifies a diseased leaf, this model classifies the specific type of disease among multiple possible classes.

The supported rice leaf diseases include:

- Bacterial Blight  
- Blast  
- Brown Spot  
- Tungro  
- Hispa  
- Health (healthy leaf class)

---


## Dataset Preparation 

The dataset is collected and combined from multiple sources on Kaggle:

- [1. Rice Leaf Dataset(Train, Test & Valid)](https://www.kaggle.com/datasets/maimunulkjisan/rice-leaf-datasettrain-test-and-valid/data)  
- [2. Rice Leafs](https://www.kaggle.com/datasets/shayanriyaz/riceleafs) (only the Healthy category is used from this dataset)  
- [3. Rice Leaf Diseases Dataset](https://www.kaggle.com/datasets/raihan150146/rice-leaf-diseases-dataset)  
- [4. Leaf Disease Dataset (combination)](https://www.kaggle.com/datasets/asheniranga/leaf-disease-dataset-combination)

Steps to prepare the dataset:

1. Download the datasets from the links above
2. Combine dataset **[Rice Leaf Dataset(Train, Test & Valid)](https://www.kaggle.com/datasets/maimunulkjisan/rice-leaf-datasettrain-test-and-valid/data), [Rice Leafs](https://www.kaggle.com/datasets/shayanriyaz/riceleafs)** and organize them into folders by disease categories. **Note: in the [3. Rice Leaf Diseases Dataset](https://www.kaggle.com/datasets/raihan150146/rice-leaf-diseases-dataset), only the Healthy Rice Leaf category is utilized to represent healthy rice leaves.** The resulting merged dataset is referred to as Dataset 2.
3. Replace the **rice** category in the **[Leaf Disease Dataset (combination)](https://www.kaggle.com/datasets/asheniranga/leaf-disease-dataset-combination)** with Dataset 2 and this dataset is referred to as Dataset 1.
4. Use `Data_preprossing_1.ipynb` and `Data_preprossing_2.ipynb` to finish the data preprocessing process.



---

## Code Attribution


In the **Stage_2.ipynb** notebook, the function `prepare_for_training()` was adapted from a Kaggle notebook by Amy Jang, titled [“TensorFlow Pneumonia Classification on X-rays”](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays/notebook#9.-Predict-and-evaluate-results).

This function is used to prepare the dataset for training by implementing efficient shuffling and caching mechanisms. While it has been slightly modified to fit the needs of this project, the core structure remains true to the original implementation.

Full credit goes to the original author and the Kaggle community for providing helpful open-source resources.

---

## Project Structure

```
.
├── Data_preprossing_1.ipynb               # Data preprocessing for Dataset 1
├── requirement.txt                        # The file for building the environment
├── Data_preprossing_2.ipynb               # Data preprocessing for Dataset 2
├── Stage_1.ipynb                          # Stage 1 binary classification model training
├── Stage_2.ipynb                          # Stage 2 multiple classification model training）
├── stage_1_model-MobileNet.weights.h5     # Stage 1 MobileNetV1 weights (not included)
├── Stage_2_model-MobileNet.weights.h5     # Stage 2 MobileNetV1 weights (not included)
├── MobileNet_stage_1_logs.json            # Stage 1 Training log（MobileNetV1）(not included)
├── Stage_2_mobilenet_logs.json            # Stage 2 Training log（MobileNetV1）(not included)
├── UI/
│   └── User_Interface.py                  # Gradio based User interface
└── README.md                              # This file

```


> **Note**: Weights (`.h5`) and training log (`.json`) of different models are large and not stored in the repo. You must download or place them manually from github!

---

## Features

- ✅ Balance Between Performance and Efficiency
- ✅ Two-Stage Deep Learning Classification  
- ✅ User-Friendly Interface via Gradio  
- ✅ Confidence Bar Chart for Stage 2 Predictions  
- ✅ Modular Design for Custom Training  
- ✅ Lightweight MobileNet Architecture  

---

## Setup and Installation

### 1. **Clone the Repository**

```bash
git clone https://github.com/Jack-622/Rice-Leaf-Disease-Classification.git
cd rice-leaf-disease-classification
```

### 2. **Create a Python 3.9 Virtual Environment**

```bash
# macOS / Linux
python3.9 -m venv venv
source venv/bin/activate

# Windows
py -3.9 -m venv venv
venv\Scripts\activate
```

### 3. **Install Required Libraries**

```bash
pip install -r requirement.txt
```

### 4. **Place the Model Files**  
   Put `stage_1_model-MobileNet.weights.h5` and `Stage_2_model-MobileNet.weights.h5` in the same folder as `Stage_1.ipynb`and `Stage_2.ipynb`.

> ⚠️ **Important:** Make sure to update the file paths in all related scripts and notebooks (e.g., `User_Interface.py`, `Stage_1.ipynb`, `Stage_2.ipynb`, `Data_preprocessing_1` and `Data_preprocessing_2`) to correctly point to the model files if you place them in a different directory.
---

## Usage

Run the following command to launch the interface:

```bash
python User_Interface.py
```

A local Gradio web link will be shown in your terminal (e.g., http://127.0.0.1:7860).  
And a public Gradio web link will be shown in your terminal, you can use it in your phone and other devices.   
Open it in your browser, upload a rice leaf image, and view the classification results.

---

## How It Works

1. **Preprocessing**
   - Resize to 224x224
   - Normalize pixel values to `[0, 1]`

2. **Stage 1**: Binary classification (`Diseased` or `Not Diseased`)

3. **Stage 2**: If diseased, classify into one of:
   - Bacterial Blight  
   - Blast  
   - Brown Spot  
   - Tungro  
   - Hispa  
   - Health

4. **Output**:
   - Text Summaries with their probabilities (Stage 1 + Stage 2 results)
   - Probability bar chart of all class predictions

---

## Utilized Model Details

- **Architecture**: VGG-16, DenseNet-121, ResNet-50, MobileNetV1 and EfficientNetB0 with Modified Fully Connected Layer
- **Loss Functions**:
  - Stage 1: Binary Crossentropy  
  - Stage 2: Categorical Crossentropy  
- **Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC  
- **Optimizers**: Adam/SGD/RMSprop  

---

## Notebooks Explanation
- `Data_preprossing_1.ipynb`: Data preprocessing process for Dataset 1  
- `Data_preprossing_2.ipynb`: Data preprocessing process for Dataset 2 
- `Stage_1.ipynb`: Train and evaluate Stage 1 binary model  
- `Stage_2.ipynb`: Train and evaluate Stage 2 multi-classification model  
- `User_Interface.py`: Gradio GUI interface for model inference

---

## Contributing

1. Fork the repo  
2. Create a branch: `git checkout -b feature/your-feature`  
3. Commit your changes: `git commit -m 'Add feature'`  
4. Push to GitHub: `git push origin feature/your-feature`  
5. Open a Pull Request!

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contact

**Maintainer**: Jiajun Mai  
**Email**: 1064354741@qq.com / 202116801522@cdu.edu.cn    
**GitHub**: [https://github.com/Jack-622](https://github.com/Jack-622)

---

**Happy Plant Diagnosing! 🌾**
