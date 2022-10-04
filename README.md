# Credit_Risk_Analysis
## Overview
In this project, we are going to apply machine learning to solve a real-world challenge: credit card risk prediction. We’ll use the credit card dataset from LendingClub to evaluate the performance of several machine learning models and make a written recommendation on whether they should be used to predict credit risk.

## Resources
Data Source: LoanStats_2019Q1.csv<br/>
Software: Python 3.7.6 ,  Jupyter Notebook 6.4.8

## Results
### Naive Random Oversampling (RandomOverSampler)
![RandomOverSampler](https://user-images.githubusercontent.com/107179765/193948494-3f17f4d4-ef00-46fe-a4d5-8d17383607ed.png)

### SMOTE Oversampling (SMOTE)
![SMOTE](https://user-images.githubusercontent.com/107179765/193948518-91ee1988-1f1b-4be3-8e5d-b1e9faee80a5.png)

### Undersampling (ClusterCentroids)
![ClusterCentroids](https://user-images.githubusercontent.com/107179765/193948528-133e1db7-58cb-448e-9c1a-b9815e91a5fb.png)

### Combination(Over and Under) Sampling (SMOTEENN)
![SMOTEENN](https://user-images.githubusercontent.com/107179765/193948539-5df7cb5d-8430-41dc-a508-9aad99000769.png)

### Balanced Random Forest Classifier (BalancedRandomForestClassifier)
![BalancedRandomForestClassifier](https://user-images.githubusercontent.com/107179765/193948555-fd88c178-7c1d-42c8-a7ac-a91cdb06c4a1.png)

### Easy Ensemble AdaBoost Classifier (EasyEnsembleClassifier)
![EasyEnsembleClassifier](https://user-images.githubusercontent.com/107179765/193948573-15771565-baf6-4bcf-a7fc-6fea3904539f.png)

