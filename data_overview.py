import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_data = pd.read_csv('train.csv', sep=',', encoding='utf-8')

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    # General Information
    print('===  Dataframe Head  ===')
    print(df_data.head())

    print()
    print('===  Dataframe Describe  ===')
    print(df_data.describe)

    print()
    print('===  Dataframe Info  ===')
    print(df_data.info())

    print()
    print('===  Dataframe Size  ===')
    print(df_data.size)

    print()
    print('===  Dataframe Shape  ===')
    print(df_data.shape)

    print()
    print('===  Dataframe Number of Rows  ===')
    df_data_row_number = df_data.shape[0]
    print(df_data_row_number)

    print()
    print('===  Dataframe Number of Columns  ===')
    df_data_column_number = df_data.shape[1]
    print(df_data_column_number)

    print()
    print('===  Data Headers  ===')
    df_data_headers = list(df_data)
    print(df_data_headers)
    print()

    # Number of unique values per column
    print('===  Number of Unique User Ids  ===')
    df_data_unique_user_number = df_data['user_id'].nunique()
    print(df_data_unique_user_number)
    print('Percentage of unique User Ids: ', round((df_data_unique_user_number / df_data_row_number * 100), 2), '%')
    print()

    print('===  Number of Unique Session Ids  ===')
    df_data_unique_session_number = df_data['session_id'].nunique()
    print(df_data_unique_session_number)
    print('Percentage of unique Session Ids: ', round((df_data_unique_session_number / df_data_row_number * 100), 2),
          '%')
    print()

    print('===  Number of Click Outs  ===')
    df_data_click_out_number = df_data[df_data['action_type'] == 'clickout item'].shape[0]
    print(df_data_click_out_number)
    print('Percentage of Click Outs: ', round((df_data_click_out_number / df_data_row_number * 100), 2), '%')
    print()

    print('===  Calculate percentage of Sessions with Click Out  ===')
    # create df with session_id and True/False if action_type == clickout item
    data_sessions_type = [df_data['session_id'], df_data['action_type'] == 'clickout item']
    data_sessions_type_headers = ['session_id', 'action_type']
    df_data_sessions_type = pd.concat(data_sessions_type, axis=1, keys=data_sessions_type_headers)
    df_data_sessions_type = df_data_sessions_type.drop_duplicates()
    # count number of sessions per True/False
    df_data_sessions_per_true_false = df_data_sessions_type['action_type'].value_counts()
    # get only True
    df_data_sessions_type_unique_clickout_sessions_number = df_data_sessions_per_true_false.loc[True]
    print(round((df_data_sessions_type_unique_clickout_sessions_number / df_data_unique_session_number) * 100, 2), '%')
    print('Number of sessions with clickout: ', df_data_sessions_type_unique_clickout_sessions_number)
    print()

    print('===  Missing values per column  ===')
    print(df_data.isnull().sum())
    print()

    print('===  Top 10 Cities With High Activity  ===')
    df_data_ten_most_common_cities = df_data['city'].value_counts()[:10]
    print(df_data_ten_most_common_cities)
    df_data_ten_most_common_cities.plot.bar()
    plt.show()
    print()

    print('=== Most Common Current Filters  ===')
    # Most common filters
    df_data_most_common_current_filters = df_data['current_filters'].value_counts()
    most_common_current_filter = df_data_most_common_current_filters.index[0]
    print(df_data_most_common_current_filters)

    print()
    print('Most common filter: ', most_common_current_filter)
    print()

    print('===  Top 10 Filters  ===')
    df_data_top_ten_most_common_filters = df_data['current_filters'].value_counts()[:10]
    print(df_data_top_ten_most_common_filters)
    df_data_top_ten_most_common_filters.plot.bar()
    plt.show()

    print()
    print('===  Action Type Analysis  ===')
    df_data_action_type = df_data['action_type'].value_counts()
    print(df_data_action_type)
    df_data_action_type.plot.bar()
    plt.show()

    print()
    print('===  Top 10 References  ===')
    df_data_references = df_data['reference'].value_counts()[:10]
    print(df_data_references)
    df_data_references.plot.bar()
    plt.show()

    print()
    print('===  Top 10 Platforms  ===')
    df_data_platforms = df_data['platform'].value_counts()[:10]
    print(df_data_platforms)
    df_data_platforms.plot.bar()
    plt.show()

    print()
    print('===  Device Analysis  ===')
    df_data_device = df_data['device'].value_counts()
    print(df_data_device)
    df_data_device.plot.bar()
    plt.show()
