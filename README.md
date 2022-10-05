# Credit_Risk_Analysis
## Overview
In this project, we are going to apply machine learning to solve a real-world challenge: credit card risk prediction. We’ll use the credit card dataset from LendingClub to evaluate the performance of several machine learning models and make a written recommendation on whether they should be used to predict credit risk.

## Resources
Data Source: LoanStats_2019Q1.csv<br/>
Software: Python 3.7.6 ,  Jupyter Notebook 6.4.8

## Results
### Naive Random Oversampling (RandomOverSampler)
- The balanced accuracy score is 0.64.<br/>
- The precision and recall scores of high_risk is 0.01 and 0.66, the precision and recall scores of low_risk is 1.00 and 0.62.<br/>
- The f1 score of high_risk is 0.02, while the f1 score of low_risk is 0.76.<br/>
![RandomOverSampler](https://user-images.githubusercontent.com/107179765/193948494-3f17f4d4-ef00-46fe-a4d5-8d17383607ed.png)

### SMOTE Oversampling (SMOTE)
- The balanced accuracy score is 0.54.<br/>
- The precision and recall scores of high_risk is 0.01 and 0.61, the precision and recall scores of low_risk is 1.00 and 0.69.<br/>
- The f1 score of high_risk is 0.02, while the f1 score of low_risk is 0.81.<br/>
![SMOTE](https://user-images.githubusercontent.com/107179765/193948518-91ee1988-1f1b-4be3-8e5d-b1e9faee80a5.png)

### Undersampling (ClusterCentroids)
- The balanced accuracy score is 0.65.<br/>
- The precision and recall scores of high_risk is 0.01 and 0.69, the precision and recall scores of low_risk is 1.00 and 0.40.<br/>
- The f1 score of high_risk is 0.01, while the f1 score of low_risk is 0.57.<br/>
![ClusterCentroids](https://user-images.githubusercontent.com/107179765/193948528-133e1db7-58cb-448e-9c1a-b9815e91a5fb.png)

### Combination(Over and Under) Sampling (SMOTEENN)
- The balanced accuracy score is 0.64.<br/>
- The precision and recall scores of high_risk is 0.01 and 0.71, the precision and recall scores of low_risk is 1.00 and 0.57.<br/>
- The f1 score of high_risk is 0.02, while the f1 score of low_risk is 0.73.<br/>
![SMOTEENN](https://user-images.githubusercontent.com/107179765/193948539-5df7cb5d-8430-41dc-a508-9aad99000769.png)

### Balanced Random Forest Classifier (BalancedRandomForestClassifier)
- The balanced accuracy score is 0.79.<br/>
- The precision and recall scores of high_risk is 0.03 and 0.70, the precision and recall scores of low_risk is 1.00 and 0.87.<br/>
- The f1 score of high_risk is 0.06, while the f1 score of low_risk is 0.93.<br/>
![BalancedRandomForestClassifier](https://user-images.githubusercontent.com/107179765/193948555-fd88c178-7c1d-42c8-a7ac-a91cdb06c4a1.png)

### Easy Ensemble AdaBoost Classifier (EasyEnsembleClassifier)
- The balanced accuracy score is 0.93.<br/>
- The precision and recall scores of high_risk is 0.09 and 0.92, the precision and recall scores of low_risk is 1.00 and 0.94.<br/>
- The f1 score of high_risk is 0.16, while the f1 score of low_risk is 0.97.<br/>
![EasyEnsembleClassifier](https://user-images.githubusercontent.com/107179765/193948573-15771565-baf6-4bcf-a7fc-6fea3904539f.png)

From the outputs above, we can see that:<br/>
- All the models used to perform the credit card risk analysis show weak precision in determining if a credit card risk is high. The Ensemble models brought a lot more improvements especially on the sensitivity of the high_risk predicition.<br/>
- With a precision of 1 for low_risk while a precision of less than 0.10 for high_risk, which means the model is more reliable when predict low_risk, but there were lot of mis-prediction on high_risk.<br/>
- Apparently, the EasyEnsembleClassifier model shows higher balanced accuracy score, precision and recall scores than other models, which means it’s a more reliable model than other models.<br/>
- As for the high_risk, we want high sensitivity score since we want all high_risk can be predicted out and less false negative. As for the low_risk, we want high precision score since we want less high predicted to be low_risk.<br/>

## Summary
The results show that 
