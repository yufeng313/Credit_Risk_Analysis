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
- The balanced accuracy score is 0.65.<br/>
- The precision and recall scores of high_risk is 0.01 and 0.61, the precision and recall scores of low_risk is 1.00 and 0.69.<br/>
- The f1 score of high_risk is 0.02, while the f1 score of low_risk is 0.81.<br/>
![SMOTE](https://user-images.githubusercontent.com/107179765/193948518-91ee1988-1f1b-4be3-8e5d-b1e9faee80a5.png)

### Undersampling (ClusterCentroids)
- The balanced accuracy score is 0.54.<br/>
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
- Overall, all the models used to perform the credit card risk analysis show weak precision in determining if a credit card risk is high. The Ensemble models brought a lot more improvements especially on the sensitivity of the high-risk prediction. <br/>
- With a precision of 1.00 for low-risk while a precision of less than 0.10 for high-risk, which means the model is more reliable when predict credit card to be low-risk, but there were a lot of low-risk predicted to be high-risk by mistake. <br/>
- Apparently, the EasyEnsembleClassifier model shows the highest balanced accuracy score, precision and recall scores than other models, which means it’s a more reliable model than other models. <br/>
- In this analysis, high sensitivity for high-risk is more important than high precision for a credit card risk prediction since we want all high-risk can be predicted out and less false negative. As for the low-risk, we want higher precision score than sensitivity score since we want less truly high-risk mis-predicted to be low-risk. <br/>

## Summary
In this project, we use different machine learning models to predict credit risk:<br/>
1) Oversample the data using the RandomOverSampler and SMOTE algorithms, <br/>
2) Undersample the data using the ClusterCentroids algorithm, <br/>
3) Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm,<br/>
4) Compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. <br/>

The results show that all the models used to perform the credit card risk analysis show weak precision in determining if a credit card risk is high. The Ensemble models brought a lot more improvements especially on the sensitivity of the high-risk prediction.<br/>
But when we took a deep look from the most reliable model - the EasyEnsembleClassifier model – with a sensitivity of 0.92 means it detects almost all the high-risk credit card, and a precision of 0.09 means a lot of low-risk were predicted to be high-risk mistakenly, which will lead the bank spending a lot of time on eliminating those false positive predictions. Also with a low f1 score of 0.16, clearly, this is not a useful algorithm. So, I would not recommend any of the models for the bank.
