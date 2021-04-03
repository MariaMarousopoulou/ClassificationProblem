import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import df_data
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression


def print_function(print_text):
    if __name__ == '__main__':
        print(print_text)
    return


def print_feature_importance(feat_importance):
    if __name__ == '__main__':
        for feature, importance in enumerate(feat_importance):
            print('Feature', feature, '=>  Importance:', importance)
    return


def plot_feature_importance(feat_importance):
    if __name__ == '__main__':
        plt.bar([x for x in range(len(feat_importance))], feat_importance)
        plt.title('Feature Importance')
        plt.show()
    return


# Separate data and classes
df_labels = df_data['action_type']
df_data = df_data.drop('action_type', axis=1)


if __name__ == '__main__':
    print_function('===  Feature Evaluation  ===')

    print_function('===  PCA  ===')
    pca = PCA(n_components=6)
    pca_data = pca.fit(df_data).transform(df_data)
    print_function('Percentage of variance explained for each components: %s' % str(pca.explained_variance_ratio_))

    print_function('===  Linear Regression  ===')
    linear_regression = LinearRegression()
    linear_regression.fit(df_data, df_labels)
    linear_regression_feature_importance = linear_regression.coef_
    print_feature_importance(linear_regression_feature_importance)
    plot_feature_importance(linear_regression_feature_importance)

    print_function('===  Logistic Regression  ===')
    logistic_regression = LogisticRegression()
    logistic_regression.fit(df_data, df_labels)
    logistic_regression_feature_importance = logistic_regression.coef_[0]
    print_feature_importance(logistic_regression_feature_importance)
    plot_feature_importance(logistic_regression_feature_importance)

    print_function('===  CART Regression  ===')
    cart_regression = DecisionTreeRegressor()
    cart_regression.fit(df_data, df_labels)
    cart_regression_feature_importance = cart_regression.feature_importances_
    print_feature_importance(cart_regression_feature_importance)
    plot_feature_importance(cart_regression_feature_importance)

    print_function('===  KBest  ===')
    kbest = SelectKBest(score_func=chi2, k=6)
    kbest_fit = kbest.fit(df_data, df_labels)
    kbest_dfscores = pd.DataFrame(kbest_fit.scores_)
    kbest_dfcolumns = pd.DataFrame(df_data.columns)
    kbest_df = pd.concat([kbest_dfcolumns, kbest_dfscores], axis=1)
    kbest_df.columns = ['Features', 'Score']
    print_function(kbest_df)


# Split data into train test set
print_function('===  Split Data into Train & Test Set  ===')
df_data_train, df_data_test, df_labels_train, df_labels_test = train_test_split(df_data, df_labels, test_size=0.2,
                                                                                random_state=7)
