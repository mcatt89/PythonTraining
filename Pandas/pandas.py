#Notes and samples from W3Schools Pandas Tutorial
#https://www.w3schools.com/python/pandas/default.asp 

#Pandas is for analyzing, cleaning, exploring & manipulating data in data sets.
#Pandas name comes from Panel Data & Python Data Analysis

#See environemnt.md for installing the module

import pandas
import pandas as pd

#organize array into a dataframe
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}
myvar = pandas.DataFrame(mydataset)
print(myvar)
#in display, index is added

#switch to alias
pdvar = pd.DataFrame(mydataset)
print(pdvar)
# Note: Pandas uses some capitol letters

#check version
print(pd.__version__)


# Pandas Series
#A series is a one-demensional array holding any data type

#Create a simple Pandas Series from list
p_list = [1, 7, 2]
p_series = pd.Series(p_list)
print(p_series)

#Review Index labels of the series
print(p_series)

#Add labels to the index
p_labeledSeries = pd.Series(p_list, index = ["x", "y", "z"])
print(p_labeledSeries)

#Access item by referencing label
print(p_labeledSeries["y"])
#label/index is not displayed, only the element in the array


#Create pandas series from a dictionary
calories = {"day1": 420, "day2": 380, "day3": 390}
c_series = pd.Series(calories)
print(c_series)

#create a series from calories, but only for day1 & day2
c_series2 = pd.Series(calories, index = ["day1", "day2"])
print(c_series2)


# Data Frames

#Series are one-demensional (single column) DataFrames are 2-demensional and show a whole table
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data)
print(df)

# locate a row in the data frame with the index
print(df.loc[0])

#print the first two rows
print(df.loc[[0, 1]])

#add named indexes to the data frame
df_named = pd.DataFrame(data, index = ["day1", "day2", "day3"])
print(df_named)

#access row by name
print(df_named.loc["day1"])
#when the index is named, it can no longer be referenced by number
print(df_named.loc[0])


# Read CSV
#.read_csv() always loads into data frame

#import data from csv
csv_data = pd.read_csv('Pandas\\data.csv')
print(csv_data)

#print to string to sho all the data
print(csv_data.to_string())

#check the number of maximum returned rows
print(pd.options.display.max_rows)
#default 60

#change max rows returned
pd.options.display.max_rows = 9999
print(csv_data)


# Reading JSON

#Load JSON from file into data frame
json_data = pd.read_json('Pandas\\data.json')
print(json_data.to_string())

#JSON objects use the same format as Python dictionaries
#if the json is in your code, you can load it directly

json_incode = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df_json = pd.DataFrame(json_incode)
print(df_json)


# Analyzing Data Frames

# Head() method provides an overview of the data, starting at the top
#import data from csv
csv_data = pd.read_csv('Pandas\\data.csv')
print(csv_data.head(10))

#by default head() displays 5 rows
print(csv_data.head())

# tail() method is the same as head, but from the bottom up
print(csv_data.tail())

# info() method provides information about the data 
print(csv_data.info())


# Cleaning Data

#import data to be cleaned from csv
orig_data = pd.read_csv('Pandas\\orig_data.csv')
print(orig_data)

# Create a new data frame and remove rows with empty cells
noEmpty_data = orig_data.dropna()
print(noEmpty_data)
#The index numbers for the rows are not changed, so 31 still appears as the last row
#The rows are removed though, check for row 18 in the original and noEmpty data

#Import the data and modify the data frame to remove rows with NULL values
noEmpyt_inplace = pd.read_csv('Pandas\\orig_data.csv')
noEmpyt_inplace.dropna(inplace=True)
print(noEmpyt_inplace)

#Replace all empty values with a default value
replaceAll_data = pd.read_csv('Pandas\\orig_data.csv')
replaceAll_data.fillna(130, inplace=True)
print(replaceAll_data)

#replace emapty values in a specified column with a default value
replaceColumn_data = pd.read_csv('Pandas\\orig_data.csv')
replaceColumn_data["Calories"].fillna(130, inplace=True)
print(replaceColumn_data)

#Replace empty values in a column with the mean from the column
replaceMean_data = pd.read_csv('Pandas\\orig_data.csv')
c_mean = replaceMean_data["Calories"].mean()
replaceMean_data["Calories"].fillna(c_mean, inplace=True)
print(replaceMean_data)

#Replace empty values in a column with the median value (median is the middle value after sorting valued ascending)
replaceMedian_data = pd.read_csv('Pandas\\orig_data.csv')
c_median = replaceMedian_data["Calories"].median()
replaceMedian_data["Calories"].fillna(c_median, inplace = True)
print(replaceMedian_data)

#Replace empty values in a column with the mode value (mode is the most commonly appearing value)
replaceMode_data = pd.read_csv('Pandas\\orig_data.csv')
c_mode = replaceMode_data["Calories"].mode()[0] # the [0] selects the first value to appear if there is a tie in most commonly appearing values
replaceMode_data["Calories"].fillna(c_mode, inplace=True)
print(replaceMode_data)
print(c_mode)


#Start with clearing wrong format https://www.w3schools.com/python/pandas/pandas_cleaning_wrong_format.asp 



