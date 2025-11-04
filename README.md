# Quaternion Matrix Factorization for Movie Recommendations 

This project implements a **Quaternion-based Matrix Factorization (QMF)** model for collaborative filtering using **PyTorch**.  
Unlike traditional vector embeddings, quaternion embeddings capture richer user–item interactions in a 4D hypercomplex space.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Description](#model-description)
- [Training Results](#training-results)
- [Example Output](#example-output)
- [Key Functions](#key-functions)

---

## Overview

Matrix Factorization (MF) is widely used in recommender systems.  
This project extends MF using **quaternion embeddings**, which generalize real and complex numbers into four dimensions.  
Each user and item is represented by a quaternion, and their interaction is computed through quaternion multiplication.

The model supports two projection methods:
- **Radius projection:** computes the magnitude of the quaternion.  
- **Angle projection:** computes angular relationships between quaternions.

---

## Project Structure

```bash
│
├── quaternion_matrix_factorization.py   # Main training and recommendation script
├── README.md                            # Project documentation
└── ml-100k/                             # MovieLens 100K dataset
     ├── u.data
     └── u.item
```
## Requirements

Install the dependencies using:

```bash
pip install numpy pandas torch tqdm
```

## Dataset

The model uses the **MovieLens 100K** dataset, which is a popular benchmark for recommender systems.  
It can be downloaded from the official GroupLens website:

📦 **Download here:**  
[https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

### Dataset Description

- **u.data** — Contains user–item interactions in the form of `user_id`, `item_id`, `rating`, and `timestamp`.  
- **u.item** — Contains movie information such as `movie_id` and `title`.

### Folder Structure

After extracting the dataset, the structure should look like this:

```bash
ml-100k/
├── u.data
└── u.item

```
### Update File Paths in Code

Make sure to update the file paths in your Python script before running it.
Example:

``` bash
data_path = "/home/<username>/Downloads/ml-100k/u.data"
movie_path = "/home/<username>/Downloads/ml-100k/u.item"
```

### Loading the Dataset
```bash
import pandas as pd

# Define column names
columns = ['user_id', 'item_id', 'rating', 'timestamp']

# Load rating data
df = pd.read_csv(data_path, sep='\t', names=columns)

# Load movie titles
movies_df = pd.read_csv(movie_path, sep='|', encoding='latin-1',
                        header=None, usecols=[0, 1], names=['movie_id', 'title'])

print(df.head())
print(movies_df.head())


```
## How to Run

Ensure the dataset paths are correct in the code.

### **Run the script:**
```bash
python quaternion.ipynb
```

### **The script will:**

- Train the model using different quaternion embedding sizes (K = 4, 8, 16, 32)

- Compare both **Radius** and **Angle** projection methods

- Report **RMSE** and **MAE** metrics

- Generate **Top-N movie recommendations** for a sample user


## Model Description
### Quaternion Matrix Factorization (QMF)

A quaternion is represented as:
```bash
q = a + bi + cj + dk
```

where a is the real part and b, c, d are imaginary components.

In this model:

- Each user and item has a 4K-dimensional quaternion embedding.

- User–item interaction is computed using the Hamilton product between embeddings.

- The result is projected via either:

  * Radius projection: magnitude of quaternion.
  
  * Angle projection: angular component of quaternion.

## Training Results

### Quaternion Embedding Performance

| **Type**      | **K** | **Projection** | **RMSE** | **MAE** |
|----------------|-------|----------------|-----------|----------|
| Quaternion     | 4     | radius         | 0.5619    | 0.4823   |
| Quaternion     | 4     | angle          | 1.1969    | 1.0590   |
| Quaternion     | 8     | radius         | 0.5558    | 0.4754   |
| Quaternion     | 8     | angle          | 1.1907    | 1.0575   |
| Quaternion     | 16    | radius         | 0.5476    | 0.4664   |
| Quaternion     | 16    | angle          | 1.1782    | 1.0470   |
| Quaternion     | 32    | radius         | **0.5369** | **0.4549** |
| Quaternion     | 32    | angle          | 1.1555    | 1.0240   |

#### 🏆 Best Result
Embedding size **K = 32** using **Radius projection** achieved the lowest RMSE and MAE.



## Example Output

### Training Logs (K=32, Radius Projection)
```bash
Epoch 1: Loss = 0.2950, RMSE = 0.5431, MAE = 0.4615
Epoch 5: Loss = 0.2921, RMSE = 0.5405, MAE = 0.4587
Epoch 10: Loss = 0.2883, RMSE = 0.5369, MAE = 0.4549
```
Top 5 Recommendations for User 1
```bash
Top 5 Recommendations for User 1:
1. Get Shorty (1995)
2. Copycat (1995)
3. GoldenEye (1995)
4. Toy Story (1995)
5. Four Rooms (1995)
```
---
## Key Functions

| **Function** | **Description** |
|---------------|-----------------|
| `QuaternionMatrixFactorization` | Defines the quaternion-based user and item embedding model. |
| `train_model()` | Trains the model using MSE loss and the AdamW optimizer. |
| `predict_full_matrix()` | Computes the full user–item prediction matrix. |
| `recommend_top_n()` | Generates top-N movie recommendations for a given user. |
| `normalize_ratings()` | Normalizes rating values to the range [0, 1]. |

---
## License
Educational project - All rights reserved.


