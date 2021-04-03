from data_overview import df_data
from sklearn.preprocessing import LabelEncoder


def print_function(print_text):
    if __name__ == '__main__':
        print(print_text)
    return


print_function('===  Drop User Ids  ===')
df_data = df_data.drop('user_id', axis=1)

print_function('===  Drop Session Ids  ===')
df_data = df_data.drop('session_id', axis=1)

print_function('===  Drop Timestamp  ===')
df_data = df_data.drop('timestamp', axis=1)

print_function('=== Replace Action Type  ===')
print_function('0: Not clickout item')
df_data['action_type'].replace('interaction item image', 0, inplace=True)
df_data['action_type'].replace('filter selection', 0, inplace=True)
df_data['action_type'].replace('search for destination', 0, inplace=True)
df_data['action_type'].replace('change of sort order', 0, inplace=True)
df_data['action_type'].replace('interaction item info', 0, inplace=True)
df_data['action_type'].replace('interaction item rating', 0, inplace=True)
df_data['action_type'].replace('interaction item deals', 0, inplace=True)
df_data['action_type'].replace('search for item', 0, inplace=True)
df_data['action_type'].replace('search for poi', 0, inplace=True)
print_function('1: Clickout item')
df_data['action_type'].replace('clickout item', 1, inplace=True)

label_encoder = LabelEncoder()
print_function('===  Label Encoding Reference  ===')
df_data['reference'] = label_encoder.fit_transform(df_data['reference'])

print_function('===  Label Encoding Platforms  ===')
df_data['platform'] = label_encoder.fit_transform(df_data['platform'])

print_function('===  Label Encoding Cities  ===')
df_data['city'] = label_encoder.fit_transform(df_data['city'])

print_function('===  Replace Device Type  ===')
print_function('Mobile: 0')
df_data['device'].replace('mobile', 0, inplace=True)
print_function('Desktop: 1')
df_data['device'].replace('desktop', 1, inplace=True)
print_function('Tablet: 2')
df_data['device'].replace('tablet', 2, inplace=True)

print_function('===  Label Encoding Current Filters  ===')
df_data['current_filters'] = label_encoder.fit_transform(df_data['current_filters'])

print_function('===  Drop Impressions  ===')
df_data = df_data.drop('impressions', axis=1)

print_function('===  Drop Prices  ===')
df_data = df_data.drop('prices', axis=1)
