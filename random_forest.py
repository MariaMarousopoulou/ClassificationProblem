import time
from sklearn.ensemble import RandomForestClassifier
from feature_split import df_data_train, df_data_test, df_labels_train, df_labels_test
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score, \
                            classification_report


print('===  Random Forest, Estimators: 10, Criteria: Gini, Max Depth: 6  ===')
start = time.time()
# Create classifier
random_forest_gini_depth_6 = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=6)
# Train classifier
random_forest_gini_depth_6 = random_forest_gini_depth_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_random_forest_gini_depth_6 = random_forest_gini_depth_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_random_forest_gini_depth_6, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_random_forest_gini_depth_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_random_forest_gini_depth_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_random_forest_gini_depth_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_random_forest_gini_depth_6, labels=None)
print(confMatrix)
end = time.time()
print("Calculation Time in Seconds:", (end - start))


print('===  Random Forest, Estimators: 10, Criteria: Gini, Max Depth: 20  ===')
start = time.time()
# Create classifier
random_forest_gini_depth_20 = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=20)
# Train classifier
random_forest_gini_depth_20 = random_forest_gini_depth_20.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_random_forest_gini_depth_20 = random_forest_gini_depth_20.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_random_forest_gini_depth_20, average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_random_forest_gini_depth_20))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_random_forest_gini_depth_20))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_random_forest_gini_depth_20))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_random_forest_gini_depth_20, labels=None)
print(confMatrix)
end = time.time()
print("Calculation Time in Seconds:", (end - start))


print('===  Random Forest, Estimators: 50, Criteria: Entropy, Max Depth: 6  ===')
start = time.time()
# Create classifier
random_forest_entropy_depth_6 = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=6)
# Train classifier
random_forest_entropy_depth_6 = random_forest_entropy_depth_6.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_random_forest_entropy_depth_6 = random_forest_entropy_depth_6.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_6,
                                       average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_6))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_6))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_6))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_6, labels=None)
print(confMatrix)
end = time.time()
print("Calculation Time in Seconds:", (end - start))


print('===  Random Forest, Estimators: 50, Criteria: Entropy, Max Depth: 20  ===')
start = time.time()
# Create classifier
random_forest_entropy_depth_20 = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=20)
# Train classifier
random_forest_entropy_depth_20 = random_forest_entropy_depth_20.fit(df_data_train, df_labels_train)
# Test classifier
df_labels_test_pred_random_forest_entropy_depth_20 = random_forest_entropy_depth_20.predict(df_data_test)
# Evaluate classifier
print('Macro: test-Precision-Recall-FScore-Support',
      precision_recall_fscore_support(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_20,
                                      average='macro'))
print("F1 Score:", f1_score(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_20))
print("Accuracy Score:", accuracy_score(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_20))
print("Classification Report:")
print(classification_report(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_20))
print("Confusion Matrix:")
confMatrix = confusion_matrix(df_labels_test, df_labels_test_pred_random_forest_entropy_depth_20, labels=None)
print(confMatrix)
end = time.time()
print("Calculation Time in Seconds:", (end - start))
