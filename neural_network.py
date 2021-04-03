import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from feature_split import df_data_train, df_data_test, df_labels_train, df_labels_test
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score, \
    classification_report


# ROC variables
false_positive, true_positive, thresh = [], [], []


print('===  Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 6  ===')
start = time.time()
# Create classifier
ClfANN_nodes_6 = MLPClassifier(solver='adam', activation='relu', batch_size=1000, tol=1e-10, hidden_layer_sizes=(6,),
                               random_state=1, max_iter=1000, verbose=True)
# Train classifier
ANNclf_nodes_6 = ClfANN_nodes_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_ANNclf_nodes_6 = ANNclf_nodes_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_ANNclf_nodes_6, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_ANNclf_nodes_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_ANNclf_nodes_6, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_6))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_ANNclf_nodes_6 = ANNclf_nodes_6.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_ANNclf_nodes_6[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 100   ===')
start = time.time()
# Create classifier
ClfANN_nodes_100 = MLPClassifier(solver='adam', activation='relu', batch_size=100000, tol=1e-10,
                                 hidden_layer_sizes=(100,), random_state=1, max_iter=1000, verbose=True)
# Train classifier
ANNclf_nodes_100 = ClfANN_nodes_100.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_ANNclf_nodes_100 = ANNclf_nodes_100.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_ANNclf_nodes_100, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_100))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_100))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_ANNclf_nodes_100))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_ANNclf_nodes_100, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_100))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_ANNclf_nodes_100 = ANNclf_nodes_100.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_ANNclf_nodes_100[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 6, Hidden layers: 2  ===')
start = time.time()
# Create classifier
ClfANN_nodes_6_layers_2 = MLPClassifier(solver='adam', activation='relu', batch_size=1000, tol=1e-10,
                                        hidden_layer_sizes=(6, 2), random_state=1, max_iter=1000, verbose=True)
# Train classifier
ANN_nodes_6_layers_2 = ClfANN_nodes_6_layers_2.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_ANNclf_nodes_6_layers_2 = ANN_nodes_6_layers_2.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_ANNclf_nodes_6_layers_2, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_6_layers_2))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_6_layers_2))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_ANNclf_nodes_6_layers_2))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_ANNclf_nodes_6_layers_2, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_ANNclf_nodes_6_layers_2))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_ANNclf_nodes_6_layers_2 = ANN_nodes_6_layers_2.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_ANNclf_nodes_6_layers_2[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 6, Hidden layers: 100  ===')
start = time.time()
# Create classifier
ClfANN_nodes_6_layers_100 = MLPClassifier(solver='adam', activation='relu', batch_size=1000, tol=1e-10,
                                          hidden_layer_sizes=(6, 100), random_state=1, max_iter=1000, verbose=True)
# Train classifier
ANN_nodes_6_layers_100 = ClfANN_nodes_6_layers_100.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_ANN_nodes_6_layers_100 = ANN_nodes_6_layers_100.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_ANN_nodes_6_layers_100, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_ANN_nodes_6_layers_100))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_ANN_nodes_6_layers_100))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_ANN_nodes_6_layers_100))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_ANN_nodes_6_layers_100, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_ANN_nodes_6_layers_100))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_ANN_nodes_6_layers_100 = ANN_nodes_6_layers_100.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_ANN_nodes_6_layers_100[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 100, Hidden layers: 100  ===')
start = time.time()
# Create classifier
ClfANN_nodes_100_layers_100 = MLPClassifier(solver='adam', activation='relu', batch_size=1000, tol=1e-10,
                                          hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000, verbose=True)
# Train classifier
ANN_nodes_100_layers_100 = ClfANN_nodes_100_layers_100.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_ANN_nodes_100_layers_100 = ANN_nodes_100_layers_100.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_ANN_nodes_100_layers_100, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_ANN_nodes_100_layers_100))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_ANN_nodes_100_layers_100))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_ANN_nodes_100_layers_100))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_ANN_nodes_100_layers_100, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_ANN_nodes_100_layers_100))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_ANN_nodes_100_layers_100 = ANN_nodes_100_layers_100.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_ANN_nodes_100_layers_100[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


# ROC curve for different Decision Trees
lwidth = 2
plt.figure(1)
plt.plot(false_positive[0], true_positive[0], color='blue',
         label='Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 6')
plt.plot(false_positive[1], true_positive[1], color='green',
         label='Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 100')
plt.plot(false_positive[2], true_positive[2], color='red',
         label='Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 6, Hidden layers: 2')
plt.plot(false_positive[3], true_positive[3], color='black',
         label='Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 6, Hidden layers: 100')
plt.plot(false_positive[4], true_positive[4], color='pink',
         label='Solver: adam, Activation: relu, Batch Size: 1000, Hidden nodes: 100, Hidden layers: 100')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Neural Networks')
plt.legend(loc="lower right")
plt.show()
