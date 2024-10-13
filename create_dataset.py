# https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts/btm5-ppia/about_data 
import pandas as pd

# Function to process data (melt and discretize)
def process_data(data, time_columns):
    # Melt the data to convert from wide to long format
    data_melted = data.melt(id_vars=['ID', 'SegmentID', 'Roadway Name', 'From', 'To', 'Direction', 'Date'], 
                            value_vars=time_columns, 
                            var_name='Time', 
                            value_name='TrafficVolume')

    # Extract day of the week from the Date column
    data_melted['DayOfWeek'] = data_melted['Date'].dt.day_name()

    # Extract the starting hour from the time range (before the hyphen) and convert to 24-hour format
    def extract_hour_24h(time_string):
        start_time = time_string.split('-')[0]
        return int(start_time.split(':')[0])
    
    data_melted['Hour'] = data_melted['Time'].apply(extract_hour_24h)

    # Drop rows with missing values in TrafficVolume
    data_melted.dropna(subset=['TrafficVolume'], inplace=True)

    # Ensure TrafficVolume is numeric to avoid issues
    data_melted['TrafficVolume'] = pd.to_numeric(data_melted['TrafficVolume'], errors='coerce')

    # Drop any remaining rows with NaN values after conversion
    data_melted.dropna(subset=['TrafficVolume'], inplace=True)

    # Discretize the traffic volume into categories (Low, Medium, High)
    data_melted['TrafficVolumeCategory'] = pd.cut(data_melted['TrafficVolume'], 
                                                  bins=[0, 100, 200, 500], 
                                                  labels=['Low', 'Medium', 'High'])

    # Convert the 'TrafficVolumeCategory' to string to ensure consistency
    data_melted['TrafficVolumeCategory'] = data_melted['TrafficVolumeCategory'].astype(str)
    
    return data_melted

def get_processed_datasets():
    # Load the dataset
    data = pd.read_csv('Traffic_Volume_Counts_20241012.csv')

    print(data['SegmentID'].unique())
    # Filter the dataset to only include SegmentID 12809 (Change this if needed)
    data_filtered = data[data['SegmentID'] == 36705]

    # Convert Date column to datetime
    data_filtered['Date'] = pd.to_datetime(data_filtered['Date'])

    # List of time columns with 24-hour format
    time_columns = ['00:00-01:00', '01:00-02:00', '02:00-03:00', '03:00-04:00',
                    '04:00-05:00', '05:00-06:00', '06:00-07:00', '07:00-08:00',
                    '08:00-09:00', '09:00-10:00', '10:00-11:00', '11:00-12:00',
                    '12:00-13:00', '13:00-14:00', '14:00-15:00', '15:00-16:00',
                    '16:00-17:00', '17:00-18:00', '18:00-19:00', '19:00-20:00',
                    '20:00-21:00', '21:00-22:00', '22:00-23:00', '23:00-00:00']

    # Split the data into training (2012-2018) and testing (2019-2020)
    train_data = data_filtered[(data_filtered['Date'] >= '2012-01-01') & (data_filtered['Date'] < '2017-01-01')]
    test_data = data_filtered[(data_filtered['Date'] >= '2017-01-01') & (data_filtered['Date'] < '2019-01-01')]

    # Process both training and testing datasets
    train_data_melted = process_data(train_data, time_columns)
    test_data_melted = process_data(test_data, time_columns)

    # Return the processed datasets
    return train_data_melted, test_data_melted

