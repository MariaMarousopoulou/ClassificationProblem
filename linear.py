import time
from sklearn import svm
from feature_split import df_data_train, df_data_test, df_labels_train, df_labels_test
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score, \
    classification_report


print('===  Linear  ===')
start = time.time()
# Create classifier
# SVCclf_linear = svm.SVC(kernel='linear', tol=1e-10, max_iter=1000, random_state=1)
SVCclf_linear = svm.LinearSVC(tol=1e-10, dual=False, random_state=1, max_iter=1000)
print('===  Linear  ===')
# Train classifier
SVCclf_linear = SVCclf_linear.fit(df_data_train, df_labels_train)
print('===  Linear  ===')
# Test classifier
df_labels_test_pred_SVCclf_linear = SVCclf_linear.predict(df_data_test)
print('===  Linear  ===')
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support for Decision tree',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_SVCclf_linear, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_SVCclf_linear))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_SVCclf_linear))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_SVCclf_linear))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_SVCclf_linear, labels=None)
print(confMatrix)
end = time.time()
print("Calculation Time in Seconds:", (end - start))
