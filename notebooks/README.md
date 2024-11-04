# helix4262 Data Exploration Notebooks

The `notebooks` directory contains experiments run on the SG-Nex datasets given, and spans the topics of data exploration, feature engineering and model selection.

## Folder Structure

```
.
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── data_merging.ipynb
│   ├── encode_nucleotide.ipynb
│   ├── model_training_pipeline.ipynb
│   ├── neuralnetwork.ipynb
│   ├── task2.ipynb
│   ├── trial_2_model.ipynb
│   ├── trial_model.ipynb
│   └── vectorization.ipynb
```

## Documentation of Notebooks

The documentation of experiments conducted over the duration spanning the project can be seen here:

- `data_merging.ipynb`: Development of ETL pipeline from dataset given in JSON and labels in CSV to produce final dataset used in training and testing 
- `encode_nucleotide.ipynb`: Development of vectorizer models to encode nucleutide sequence to test effects on model selection as a processed feature
- `model_training_pipeline.ipynb`: Development of feature engineering pipeline for the SG-Nex dataset.
- `neuralnetwork.ipynb`: Development of `ModNet` neural network using `pytorch` and testing of hyperparameters on SG-Nex dataset.
- `task2.ipynb`: Data Exploration for **Task 2: Prediction of m6A sites in all SG-NEx direct RNA-Seq samples**.
- `trial_2_model.ipynb`: Development of `scikit-learn` Decision Tree and XGBoost ensemble models, including feature engineering, hyperparameter tuning and model selection, evaluated by **ROC-AUC** and **PR-AUC**
- `trial_model.ipynb`: Development of `scikit-learn` Decision Tree and Random Forest models including feature engineering, evaluated by **Confusion Matrix**
- `vectorizer.ipynb`: Experimentation with Trigram Vectorizer and TF-IDF Vectorizer on the accuracy and precision-recall of testing on the SG-NEx dataset.