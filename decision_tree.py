import time
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve, roc_auc_score
from feature_split import df_data_train, df_data_test, df_labels_train, df_labels_test
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score, \
    classification_report


# ROC variables
false_positive, true_positive, thresh = [], [], []


print('===  Gini, Best, Max depth 6  ===')
start = time.time()
# Create classifier
treeClf_gini_best_6 = tree.DecisionTreeClassifier(criterion="gini", splitter='best', max_depth=6, random_state=0)
# Train classifier
treeClf_gini_best_6 = treeClf_gini_best_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_treeClf_gini_best_6 = treeClf_gini_best_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_treeClf_gini_best_6, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_treeClf_gini_best_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_treeClf_gini_best_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_treeClf_gini_best_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_treeClf_gini_best_6, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_treeClf_gini_best_6))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_treeClf_gini_best_6 = treeClf_gini_best_6.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_treeClf_gini_best_6[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Gini, Random, Max depth 6  ===')
start = time.time()
# Create classifier
treeClf_gini_random_6 = tree.DecisionTreeClassifier(criterion="gini", splitter='random', max_depth=6, random_state=0)
# Train classifier
treeClf_gini_random_6 = treeClf_gini_random_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_treeClf_gini_random_6 = treeClf_gini_random_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_treeClf_gini_random_6, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_treeClf_gini_random_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_treeClf_gini_random_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_treeClf_gini_random_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_treeClf_gini_random_6, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_treeClf_gini_random_6))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_treeClf_gini_random_6 = treeClf_gini_random_6.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_treeClf_gini_random_6[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Entropy, Best, Max depth 6  ===')
start = time.time()
# Create classifier
treeClf_entropy_best_6 = tree.DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=6, random_state=0)
# Train classifier
treeClf_entropy_best_6 = treeClf_entropy_best_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_treeClf_entropy_best_6 = treeClf_entropy_best_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_treeClf_entropy_best_6, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_treeClf_entropy_best_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_treeClf_entropy_best_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_treeClf_entropy_best_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_treeClf_entropy_best_6, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_treeClf_entropy_best_6))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_treeClf_entropy_best_6 = treeClf_entropy_best_6.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_treeClf_entropy_best_6[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Entropy, Random, Max depth 6  ===')
start = time.time()
# Create classifier
treeClf_entropy_random_6 = tree.DecisionTreeClassifier(criterion="entropy", splitter='random', max_depth=6,
                                                       random_state=0)
# Train classifier
treeClf_entropy_random_6 = treeClf_entropy_random_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_treeClf_entropy_random_6 = treeClf_entropy_random_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_treeClf_entropy_random_6, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_treeClf_entropy_random_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_treeClf_entropy_random_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_treeClf_entropy_random_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_treeClf_entropy_random_6, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_treeClf_entropy_random_6))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_treeClf_entropy_random_6 = treeClf_entropy_random_6.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_treeClf_entropy_random_6[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


# ROC curve for different Decision Trees
lwidth = 2
plt.figure(1)
plt.plot(false_positive[0], true_positive[0], color='blue', label='Gini, Best, Max depth 6')
plt.plot(false_positive[1], true_positive[1], color='green', label='Gini, Random, Max depth 6')
plt.plot(false_positive[2], true_positive[2], color='red', label='Entropy, Best, Max depth 6')
plt.plot(false_positive[3], true_positive[3], color='black', label='Entropy, Random, Max depth 6')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Decision Trees')
plt.legend(loc="lower right")

plt.show()

