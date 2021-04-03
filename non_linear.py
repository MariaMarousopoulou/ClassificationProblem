import time
from sklearn import svm
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_split import df_data_train, df_data_test, df_labels_train, df_labels_test
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score, \
    classification_report

'''
model = Pipeline([('scaler', StandardScaler()),
                  ('svr', SVR(kernel='linear'))])
'''

print('===  Non Linear  ===')
start = time.time()
# Create classifier
SVCclf_non_linear = svm.SVC(kernel='rbf', gamma='auto', cache_size=7000)
# Train classifier
SVCclf_non_linear = SVCclf_non_linear.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_SVCclf_non_linear = SVCclf_non_linear.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support for Decision tree',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_SVCclf_non_linear, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_SVCclf_non_linear))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_SVCclf_non_linear))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_SVCclf_non_linear))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_SVCclf_non_linear, labels=None)
print(confMatrix)
end = time.time()
print("Calculation Time in Seconds:", (end - start))
