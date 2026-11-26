# Unsupervised Machine Learning Workshop

Hands-on exploration of unsupervised learning techniques (dimensionality reduction, clustering and anomaly detection) applied to two contrasting datasets: categorical mushroom data with a validation label, and numerical credit card data with no label, used for customer segmentation.

## Features

- **Two complementary case studies** that show how the same techniques behave differently depending on the data type and on whether a reference label exists
- **Exploratory data analysis** and cleaning, including detection of hidden nulls (`?` values) and constant columns
- **Preprocessing**: One-Hot Encoding for categorical data, median imputation and standard scaling for numerical data
- **Dimensionality reduction** with PCA and t-SNE to visualize high-dimensional data in 2D
- **Supervised baseline** with Random Forest to measure how much information the data really holds
- **Clustering** with K-Means, Agglomerative Clustering, Gaussian Mixture Models and DBSCAN
- **Validation** with external metrics (ARI, NMI) when a label exists, and internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) when it does not
- **Customer segmentation** translated into business profiles and marketing strategies
- **Anomaly detection** with Isolation Forest

## Technologies

- Python 3.9
- Pandas, NumPy
- Scikit-learn
- SciPy (hierarchical clustering and dendrograms)
- Matplotlib, Seaborn
- Jupyter Notebook

## Project Structure

```
.
├── data/
│   ├── mushrooms.csv                        # 8,124 mushrooms, 22 categorical features
│   └── credit_card.csv                      # ~8,950 card holders, 17 numerical features
├── workshop-clustering-Mushrooms.ipynb      # Part 1: categorical data with label
├── workshop-clustering-creditcard.ipynb     # Part 2: numerical data without label
├── requirements.txt
└── README.md
```

| | Part 1: Mushrooms | Part 2: Credit cards |
|---|---|---|
| Data type | Categorical | Numerical |
| Key preprocessing | One-Hot Encoding | Median imputation and scaling |
| Label | Yes (validation only) | No |
| Validation | ARI / NMI | Internal metrics and interpretability |
| Data structure | Separable groups | Continuous cloud |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alvi103-png/Workshop-Unsupervised-ML.git
   cd Workshop-Unsupervised-ML
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open the notebooks with Jupyter or VS Code (selecting the `.venv` kernel) and run them in order:

1. `workshop-clustering-Mushrooms.ipynb`
2. `workshop-clustering-creditcard.ipynb`

Datasets are loaded from the `data/` folder with relative paths, so notebooks must be run from the project root.

## Results

### Part 1: Mushrooms

- One-Hot Encoding expands 21 categorical features into 115 binary columns.
- A Random Forest baseline reaches ~100% accuracy, and around 10 PCA components are enough to keep it.
- t-SNE reveals many natural subgroups, almost all pure with respect to the edible/poisonous label.
- K-Means, Agglomerative and GMM recover the label structure with **ARI ~0.6 without ever seeing it**.
- DBSCAN obtains better internal metrics but a lower ARI: well-formed clusters do not always answer the business question.
- Jaccard distance improves DBSCAN over Euclidean distance on one-hot data.

### Part 2: Credit card segmentation

- The data forms a continuous cloud: low silhouette, no clear elbow, and DBSCAN finds a single dense cluster.
- Silhouette selects **k = 3**, and K-Means outperforms Agglomerative and GMM on all internal metrics.

| Segment | Size | Profile | Strategy |
|---|---|---|---|
| VIP buyers | ~14% | High purchase volume and frequency, low cash advance, highest full payment rate | Loyalty programs, cashback, credit limit increase |
| Low activity | ~68% | Low balance, few purchases, lowest credit limit | Activation campaigns and usage incentives |
| Cash advance dependent | ~17% | Heavy cash advance use, highest balance, rarely pays in full | Risk monitoring and refinancing products |

### Key takeaway

No single algorithm or metric wins every time. The right choice of preprocessing, distance metric and validation method depends on the data type and on the problem being solved.

## Screenshots

<!-- Add key visualizations to an images/ folder and reference them here, for example:
![t-SNE mushrooms](images/tsne-mushrooms.png)
![Segment profiles heatmap](images/heatmap-segments.png)
-->

## Roadmap

- Feature engineering for credit card data (credit utilization ratio, average purchase per transaction)
- Log transformation of skewed variables before scaling
- UMAP and HDBSCAN as alternatives to t-SNE and DBSCAN
- Association rules (Apriori / FP-Growth) on the mushroom dataset

## Datasets

- [Mushroom Classification](https://www.kaggle.com/uciml/mushroom-classification) (UCI Machine Learning Repository)
- [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

## Author

**Viviana Andrango**
Developed as part of the AI & Data Analytics training at Factoría F5.
GitHub: [@alvi103-png](https://github.com/alvi103-png)
