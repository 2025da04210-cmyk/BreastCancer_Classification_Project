# BreastCancer\_Classification\_Project



\## a. Problem Statement



The objective of this project is to build and compare multiple machine learning classification models for predicting whether a breast tumor is benign or malignant.



The project uses supervised machine learning algorithms to classify cancer diagnosis based on different medical features. Multiple classification models are trained and evaluated using different performance metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).



\## b. Dataset Description



Dataset Name: Breast Cancer Wisconsin Dataset



Dataset Description:



The dataset contains medical features extracted from digitized images of breast mass samples.



The target variable is:Diagnosis



Benign (B) → 0

Malignant (M) → 1





Dataset Characteristics:



Parameter          Description

Number of Records    569

Number of Features   30

Target Variable      Diagnosis

Problem Type         Binary Classification





Features include:



\- Radius

\- Texture

\- Perimeter

\- Area

\- Smoothness

\- Compactness

\- Concavity

\- Symmetry

\- Fractal Dimension





Data preprocessing steps performed:



\- Removed unnecessary columns (id, Unnamed:32)

\- Checked missing values

\- Encoded target variable

\- Split data into training and testing datasets

\- Applied feature scaling using StandardScaler



\## c. GitHub Repository Link



GitHub Repository: https://github.com/2025da04210-cmyk/BreastCancer\_Classification\_Project



\# d. Models Used



The following machine learning classification models were implemented:



Model	        Accuracy 	AUC Score	Precision	 Recall	           F1 Score    	        MCC Score

Logistic        0.938596491	0.992724868	0.972972973     0.857142857	   0.911392405	        0.868766493

Decision        0.929824561	0.924603175	0.904761905     0.904761905	   0.904761905	        0.849206349

K-Nearest 	0.912280702	0.954695767	0.970588235     0.785714286	   0.868421053	        0.813814655

Gaussian 	0.938596491	0.993386243	1	        0.833333333	   0.909090909	        0.871489341

Random 	        0.973684211	0.992890212	1	        0.928571429	   0.962962963	        0.944154951





\## Model Performance Comparison



The above table compares the performance of all implemented models based on evaluation metrics.



Evaluation Metrics:



\- ACCURACY : Measures overall correct predictions.

\- AUC : Measures model ability to distinguish between classes.

\- PRECISION : Measures correctly predicted positive cases.

\- RECALL : Measures ability to identify actual positive cases.

\- F1 SCORE : Balance between precision and recall.

\- MCC: Measures correlation between actual and predicted classifications.



\# Model Observations





Model Name -> Observation about Model Performance

Logistic Regression ->Performs well as a baseline model because it provides good classification accuracy and works effectively on scaled .

Decision Tree -> Easy to interpret but may overfit because of its tree-based structure.

KNN -> Performance depends on the selection of K value and feature scaling. It provides reasonable results but can be affected by noise.

Naive Bayes -> Provides fast prediction and works well with simple assumptions about feature independence.

Random Forest (Ensemble) -> Provides better generalization by combining multiple decision trees and reduces overfitting.





\# Overall Winner for Your Dataset



Based on comparison of Accuracy, AUC, Precision, Recall, F1 Score, and MCC:



Best Performing Model: Random Forest





Reason:



The winning model achieved the highest overall performance across evaluation metrics and provided better classification capability for breast cancer diagnosis.

