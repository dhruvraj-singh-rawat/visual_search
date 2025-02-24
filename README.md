# Visual Image Search Engine

This project implements a visual search engine designed to retrieve visually similar images from an image collection using various computer vision techniques. Developed as coursework for the **Computer Vision and Pattern Recognition (EEE3032)** module at the **University of Surrey**.

## 📌 Overview
Digital image collections are typically searched using textual descriptions, which are limited in capturing visual characteristics. This project bridges that gap by enabling searches based purely on visual similarity.

## 📁 Dataset
- **Microsoft Research (MSRC-v2)** image dataset containing **591 images** across **20 categories**.
- **Download dataset:** [MSRC-v2 Dataset](http://download.microsoft.com/download/3/3/9/339D8A24-47D7-412F-A1E8-1A415BC48A15/msrc_objcategimagedatabase_v2.zip)

## 🔍 Techniques Implemented
- **Global Colour Histogram:** Captures color distributions for image representation.
- **Spatial Grid Descriptors:** Integrates colour histograms and texture descriptors (using Sobel filters) for spatially-informed representations.
- **Dimensionality Reduction (PCA):** Reduced descriptor dimensions to enhance retrieval speed and accuracy.
- **Distance Metrics:** Evaluated L1, L2, Chi-Squared, and Cosine similarity metrics to optimize retrieval effectiveness.

## 📊 Evaluation Metrics
- **Precision-Recall Curves** and **Area Under the Curve (AUC)** for quantitative performance analysis.


## ⚙️ Setup & Execution

### Dependencies
- Python 3.x
- NumPy
- OpenCV
- scikit-learn
- scipy
- matplotlib
- Jupyter Notebook

### 🚀 Quick Start Guide
1. Clone the repository:
```bash
git clone <repository_url>
cd visual-search-engine```

```
2. Install required libraries:

```pip install numpy opencv-python scikit-learn scipy matplotlib notebook```

3. Run experiments and visual search via Jupyter notebook:
```jupyter notebook "Coursework - Visual Search.ipynb"```

##  📈 Results
Demonstrated effective image retrieval with robust accuracy validated through comprehensive experimentation and detailed precision-recall analysis.
##  🙌 Acknowledgments
University of Surrey
Prof. Miroslaw Bober for guidance and coursework structure.


