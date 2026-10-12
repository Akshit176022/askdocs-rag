# Model Evaluation

Precision is the fraction of predicted positives that are truly positive. Recall is the fraction of actual positives that the model finds. The F1 score is the harmonic mean of precision and recall.

Accuracy can be misleading on imbalanced data. A model that always predicts the majority class may score high accuracy while being useless, so precision and recall are preferred there.

Overfitting happens when a model memorises the training data and performs poorly on new data. Underfitting happens when the model is too simple to capture the pattern, so it performs poorly on both training and test data.

Data is usually split into training, validation and test sets. The validation set is used to tune hyperparameters and the test set is touched only once for a final estimate. Cross-validation repeats the split several times to get a more stable estimate.

For retrieval systems, hit rate at k measures whether a relevant document appears in the top k results, and mean reciprocal rank rewards placing the relevant result higher.
