import csv
from collections import defaultdict
from datetime import datetime

# Initialize a dictionary to store SegmentID and their unique dates
seg_Dates = defaultdict(set)

# Read the CSV file
file_name = "Traffic_Volume_Counts_20241012.csv"  # Update with your actual file path
with open(file_name, 'r') as file:
    reader = csv.DictReader(file)

    for row in reader:
        seg_ID = row['SegmentID']
        date_str = row['Date']  
        try:
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')  
            seg_Dates[seg_ID].add(date_obj)
        except ValueError:
            print(f"Error parsing date: {date_str} for SegmentID: {seg_ID}")

# Sort segments by the number of unique dates in descending order
seg_sorted = sorted(seg_Dates.items(), key=lambda x: len(x[1]), reverse=True)

# Get the top 10 segments with the most unique dates
top_10_segments = seg_sorted[:10]

# Display the top 10 segments with their unique sorted dates
print("Top 10 segments with the most unique dates (sorted by year, month, and day):")
for seg_ID, dates in top_10_segments:
    sorted_dates = sorted(dates)  
    sorted_dates_str = [date.strftime('%Y-%m-%d') for date in sorted_dates]  
    
    print(f"SegmentID: {seg_ID}, Unique Dates Count: {len(sorted_dates_str)}")
    print(f"Unique Dates: {sorted_dates_str}\n")




