# 🛍️ Mall Customer Segmentation using K-Means and Hierarchical Clustering

This project segments mall customers into different groups based on their **Annual Income** and **Spending Score** using **unsupervised machine learning techniques**.

## 📌 Objective

To identify distinct groups of customers and gain insights into their purchasing behavior. This segmentation can help businesses in **targeted marketing strategies**.

---

## 🔍 Project Overview

- **Dataset**: Mall_Customers.csv (from Kaggle)
- **Approach**:
  - Preprocessed data and excluded irrelevant columns like `CustomerID`
  - Scaled numerical features using `StandardScaler`
  - Applied **K-Means Clustering**
    - Used the **Elbow Method** to find optimal number of clusters
    - Visualized clusters using scatter plots
  - Applied **Agglomerative Hierarchical Clustering**
    - Built a **Dendrogram** using `scipy.cluster.hierarchy`
    - Visualized hierarchical clusters
- Final result: Identified **5 distinct customer segments**

---

## 🧪 Techniques & Tools Used

| Category            | Tools & Libraries                          |
|---------------------|--------------------------------------------|
| Language            | Python                                     |
| Data Handling       | Pandas, NumPy                              |
| Visualization       | Matplotlib, Seaborn                        |
| ML Algorithms       | KMeans, AgglomerativeClustering            |
| Clustering Metrics  | WCSS (Within-Cluster Sum of Squares)       |
| Others              | SciPy for Dendrograms                      |

---

## 📊 Visualizations

- **K-Means Clusters**  
  ![image](https://github.com/user-attachments/assets/304bf59c-2ba3-4c19-8437-824e1b57bd28)
  ![image](https://github.com/user-attachments/assets/a1a04541-d99f-40a7-985b-679b4138ac46)

- **Dendrogram for Hierarchical Clustering**  
  ![image](https://github.com/user-attachments/assets/a172c0f4-40d4-40b0-8be6-7a52705b582a)
  ![image](https://github.com/user-attachments/assets/018e49b7-11bc-45a5-b3de-2b8ea12617e0)
---

## 📁 Dataset Features

| Feature          | Description                              |
|------------------|------------------------------------------|
| `CustomerID`     | Unique ID of the customer (excluded)     |
| `Gender`         | Gender of the customer                   |
| `Age`            | Age of the customer                      |
| `Annual Income`  | Annual income in thousands (k$)          |
| `Spending Score` | Score assigned by the mall (1–100)       |

> In this project, we focused only on **Annual Income** and **Spending Score** for 2D visualization.

---

## 🧠 Results

After analysis, the model segmented customers into five clusters:
- 🟥 Careful
- 🟦 Standard
- 🟩 Target
- 🟨 Sensible
- 🟪 Careless

Each group represents unique customer behavior and spending capacity.
