import os
import csv

# Define the directory containing the files
dir_path = '/vision/u/eprakash/shanghai_anomaly/training/videos/'

# Get a list of all files in the directory
file_names = os.listdir(dir_path)

# Create an empty list to hold the rows of the CSV file
rows = []

# Loop through each file name and create a row for the CSV file
for file_name in file_names:
    full_path = os.path.join(dir_path, file_name)
    row = [full_path, 0, -1]
    rows.append(row)

# Write the rows to a CSV file named "pre-train.csv"
with open('pre-train.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    writer.writerows(rows)
