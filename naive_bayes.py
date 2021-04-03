import time
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from feature_split import df_data_train, df_data_test, df_labels_train, df_labels_test
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score, \
                            classification_report


# ROC variables
false_positive, true_positive, thresh = [], [], []


print('===  Naive Bayes, Smoothing: 1e-9  ===')
start = time.time()
# Create classifier
naive_bayes_smooth_9 = GaussianNB()
# Train classifier
naive_bayes_smooth_9 = naive_bayes_smooth_9.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_naive_bayes_smooth_9 = naive_bayes_smooth_9.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_naive_bayes_smooth_9, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_naive_bayes_smooth_9))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_naive_bayes_smooth_9))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_naive_bayes_smooth_9))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_naive_bayes_smooth_9, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_naive_bayes_smooth_9))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_naive_bayes_smooth_9 = naive_bayes_smooth_9.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_naive_bayes_smooth_9[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


print('===  Naive Bayes, Smoothing: 1e-100  ===')
start = time.time()
# Create classifier
naive_bayes_smooth_100 = GaussianNB(var_smoothing=1e-100)
# Train classifier
naive_bayes_smooth_100 = naive_bayes_smooth_100.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_naive_bayes_smooth_100 = naive_bayes_smooth_100.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_naive_bayes_smooth_100, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_naive_bayes_smooth_100))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_naive_bayes_smooth_100))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_naive_bayes_smooth_100))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_naive_bayes_smooth_100, labels=None)
print(confMatrix)
print('AUC Score:', roc_auc_score(df_labels_test, df_labels_test_pred_naive_bayes_smooth_100))
end = time.time()
print("Calculation Time in Seconds:", (end - start))
roc_df_labels_test_pred_naive_bayes_smooth_100 = naive_bayes_smooth_100.predict_proba(df_data_test)
fpos, tpos, thr = roc_curve(df_labels_test, roc_df_labels_test_pred_naive_bayes_smooth_100[:, 1])
false_positive.append(fpos)
true_positive.append(tpos)
thresh.append(thr)


# ROC curve for different Decision Trees
lwidth = 2
plt.figure(1)
plt.plot(false_positive[0], true_positive[0], color='blue', label='Naive Bayes, Smoothing: 1e-9')
plt.plot(false_positive[1], true_positive[1], color='green', label='Naive Bayes, Smoothing: 1e-100')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Naive Bayes')
plt.legend(loc="lower right")
plt.show()
