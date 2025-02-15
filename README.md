# 🎭 Feature Extraction with PCA and NMF

Welcome to the **Feature Extraction** repository! This project explores dimensionality reduction techniques, specifically **Principal Component Analysis (PCA)** and **Non-Negative Matrix Factorization (NMF)**, for feature extraction and visualization.

---

## 📂 **Project Overview**

This repository demonstrates how to use **PCA** and **NMF** to extract meaningful features from high-dimensional data, such as the **Labeled Faces in the Wild (LFW)** dataset. It includes:

- **PCA**: Linear dimensionality reduction for feature extraction.
- **NMF**: Non-negative matrix factorization for parts-based representation.
- **Visualizations**: Extracted components and reconstructed images.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **mglearn**
- **NumPy**
- **Matplotlib**

---

## 📊 **Dataset**

The project uses the **Labeled Faces in the Wild (LFW)** dataset, which contains images of faces labeled with the person's name. Each image is represented as a high-dimensional feature vector.

---

## 🧠 **Key Concepts**

### 1. **Principal Component Analysis (PCA)**
- A linear dimensionality reduction technique.
- Extracts orthogonal components that capture the maximum variance in the data.
- Used for feature extraction and noise reduction.

### 2. **Non-Negative Matrix Factorization (NMF)**
- A parts-based decomposition technique.
- Represents data as a combination of non-negative components.
- Ideal for extracting interpretable features (e.g., facial parts).

---

## 🚀 **Code Highlights**

### PCA for Feature Extraction
```python
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

### Visualizing PCA Components
```python
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title(f"{i + 1}. component")
```

### NMF for Parts-Based Representation
```python
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
```

### Visualizing NMF Components
```python
fig, axes = plt.subplots(3, 5, figsize=(15, 12))
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title(f"{i + 1}. component")
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/feature-extraction.git
   cd feature-extraction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook feature_extraction.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
