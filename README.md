# Data Science / Machine Learning Projects

These are some of the projects I worked on while following the Data Scientist program at [**OpenClassrooms**](https://openclassrooms.com/en/paths/164-data-scientist) / [**CentraleSupélec**](https://www.centralesupelec.fr/en).  
They touch upon diverse ML models (regression/classification/clustering, supervised/unsupervised, deep learning, etc.) in a wide varity of fields (banking, retail, energy, agronomy), as well as DevOps skills (API, cloud infrastructure, etc.).  
Each folder will contain a dedicated README file, going into greater details regarding the specifics of the project such as the mission, the origin and characteristics of the dataset, and important steps and results of the process. But below are already short summaries of the topics, skills and most significant libraries involved for each project.

---

## Automated classification of goods (Retail)

**Mission:** Feasibility study for a retail products classification algorithm, based on descriptions and pictures.

**Keywords:** Deep Learning, Natural Language Processing (NLP), Computer Vision, Convolutional Neural Networks (CNN), Dimensionality Reduction

**Main Libraries/Tools:** Tensorflow, Keras, PIL, Scikit-learn, Gensim

**Skills:**
- Process and utilize textual data
- Process and utilize visual data
- Implement dimensionality reduction techniques
- Graphically represent high-dimensional data

---

## Business consumption prediction (Energy)

**Mission:** Predictive model to forecast energy consumption and greenhouse gas emissions of business buildings, as well as evaluate the relevance of the ENERGY STAR score.

**Keywords:** Supervised Learning, Grid Search, Cross Validation, Ensemble Learning Methods, Decision Trees

**Main Libraries/Tools:** Scikit-learn, XGBoost

**Skills:**
- Implement a supervised regression model
- Transform or design relevant features
- Fine-tune hyperparameters to achieve optimal performance
- Evaluate the performances of a regression model (R2/RMSE)

---

## Client database segmentation (Retail)

**Mission:** Segmentation of an online retailer's client database, identification of the clusters and provision of a recommended maintenance period.

**Keywords:** Clustering, Model Maintenance, 3D Data Visualization, Dimensionality Reduction, RFM Analysis (Recency-Frequency-Monetary value)

**Main Libraries/Tools:** Scikit-learn, Plotly, Yellowbrick

**Skills:**
- Implement an unsupervised classification algorithm
- Fine-tune hyperparameters to achieve optimal performance
- Evaluate the performances of a classification model (ARI)
- Maintain a model's performances above a given threshold

---

## Cloud-based data compression (Agronomy)

**Mission:** Big data environment to extract features from fruit photos, reduce dimensionality and store output online.

**Keywords:** Big Data, Cloud, Dimensionality Reduction, Deep Learning, Transfer Learning

**Main Libraries/Tools:** PySpark, Amazon Web Services (AWS), Hadoop, Keras, Tensorflow

**Skills:**
- Deploy an infrastructure in the cloud
- Process massive data quantities with distributed computing
- Implement a model resilient to drastic changes in data input size

---

## Online scoring app deployment (Banking)

**Mission:** Online app to attribute score to credit applicants based on their estimated repayment success rate, while providing details to understand and contextualize the score given.

**Keywords:** Imbalanced Data, Cross Validation, REST API, ML Model Interpretation, Evaluation Metrics

**Main Libraries/Tools:** FastAPI, Streamlit, SHAP, LightGBM, Gunicorn

**Skills:**
- Train a classification algorithm using imbalanced data
- Implement custom evaluation metrics and cost functions
- Use code versioning software to ensure model integration
- Deploy a model online using a REST API and an interactive dashboard