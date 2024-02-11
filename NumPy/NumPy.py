# Personal workbook by NumPy training from W3Schools
# https://www.w3schools.com/python/numpy/default.asp
# By Matt Herman
# Created January 28, 2024


#NOTE: Rune this in your terminal every time you resume working with examples in the file
import numpy as np 

arr = np.array([1,2,3,4,5])

print(arr)

#print(np.__version__)

print(type(arr))

#Tuple Array
t_arr = np.array((1, 2, 3, 4, 5))
print(t_arr)
print(type(t_arr))

# 0-D array with value 42
ZeroD_arr = np.array(42)
print(ZeroD_arr)

#1-D array (default array)
OneD_arr = np.array([1,2,3,4,5])
print(OneD_arr)

#2-D Array contains two subset arrays in it
TwoD_arr = np.array([[1,2,3,],[4,5,6]])
print(TwoD_arr)

#Examples with multple array dimensions
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

#create a 5 dimension array by speccifying the number of dimensions
FiveD_arr = np.array([1,2,3,4], ndmin=5)
print(FiveD_arr)
print('number of dimensions: ', FiveD_arr.ndim)


#####
#Array Indexing
#####

arr = np.array([1, 2, 3, 4])

#print first element in the array
print(arr[0])
#print second element in array
print(arr[1])
#print the sum of the third and fourth elements
print(arr[2] + arr[3])
#print the third and fourth elements
print(arr[2], '', arr[3])

#Indexing for multiple dimensions
i2d_arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(i2d_arr)
print('2nd element on the 1st row: ', i2d_arr[0,1])
print('5th element on the second row:', i2d_arr[1,4])

#Indexing for 3 dimensions
i3d_arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(i3d_arr)
print(i3d_arr[0,1,2]) #this access the first set of two arrays, then the second row, then prints the third element

#Negative indexing accesses the items from the end of the array staring with -1
ni_arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('last element from the second dimension:', ni_arr[1,-1])


#####
#Slicing Arrays
# start:end  -Starting element is included, ending element is not
# start:end:step -how often to select an element, blank selects every item in range
#####

sa_arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Slice from index 1 to index 5
print(sa_arr[1:5]) #the element with a value of '6' is at the 5th index.  This is the end, so it is not included in the slice

# Slice from element 4 to the end of the array
print(sa_arr[4:]) #when the end is left blank, it is assumes the end of the array

# Slice from the beginning to index 4 (not included)
print(sa_arr[:4]) #when the start is left blank, it assumes the first index (index 0)

#Slice from the index 3 from the end to index 1 from the end (negative indexing)
print(sa_arr[-3:-1])

# Return every other element from index 1 to 5
print(sa_arr[1:5:2])

# Return every other element for the full array
print(sa_arr[::2]) # start and end left blank to get the whole array

#slicing 2-D arrays
s2d_arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

#From the second element, slice from index 1 to index 4
print(s2d_arr[1, 1:4]) #index to the second element and then slice as normal

#from both elements, return the second index
print(s2d_arr[0:2, 2]) #both arrays should be selected, so using 0:2 to select the third array(which does not exist) as the end
print(s2d_arr[:,2]) #leaving start and end blank also works to select all arrays

#for both elements, slice index 1 to index 4, this will return a 2-D array
print(s2d_arr[0:2, 1:4])


#####
#Data Types
#####

dt_arr = np.array([1, 2, 3, 4])

#get data type of array
print(dt_arr.dtype)


udt_arr = np.array(['apple', 'banana', 'cherry'])
print(sdt_arr.dtype) # returns <U6   Less Than Undefined Six?

#create an array and define data type string
sdt_arr = np.array([1, 2, 3, 4], dtype='S')

print(sdt_arr)
print(sdt_arr.dtype) #S1 String (size) 1 

#create array with 4 bytes integer 
i4_arr = np.array([1, 2, 3, 4], dtype='i4')

print(i4_arr)
print(i4_arr.dtype) # Returns int32


#change the data type of an array
oldarr = np.array([1.1, 2.1, 3.9])

print(oldarr)
print(oldarr.dtype)

newarr = oldarr.astype('i')

print(newarr)
print(newarr.dtype) #the conversion drops the decimal from the float so they are integers.  This does not round to the closest, it is just a drop


#convert to integer with paramenter value
old2arr = np.array([1.1, 2.1, 3.1])

new2arr = arr.astype(int)

print(new2arr)
print(new2arr.dtype)


#convert to boolean
oldBarr = np.array([1, 0, 3])

newBarr = oldBarr.astype(bool)

print(newBarr) #when converted, any value not 0 is true
print(newBarr.dtype)



#####
#Copy vs View
#####

###Copy
#copy the original and make a change
source_arr = np.array([1,2,3,4,5])

#copy the source array to a new array
copy_arr = source_arr.copy()

#change the first element to 42
copy_arr[0] = 42

#print both arrays
print(source_arr)
print(copy_arr)


###View
source2_arr = np.array([1,2,3,4,5])
view_arr = source2_arr.view()

#chagne first element to 42, since this is a view, the change will affect both the view & source
view_arr[0] = 42

#print both
print(source2_arr)
print(view_arr)



####check if the array owns the data
source3_arr = np.array([1,2,3,4,5])

copy2_arr = source3_arr.copy()
view2_arr = source3_arr.view()

print(copy2_arr.base) #returns 'None' which means this array owns the data
print(view2_arr.base) #returns the original array which means this array does not own the data


#####
#Shape of an Array
#####

shape_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(shape_arr.shape)

shape2_arr = np.array([1,2,3,4], ndmin=5)
print(shape2_arr)
print('shape of array: ', shape2_arr.shape)

#reshape array
shape3_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
reshape_arr = shape3_arr.reshape(2,2,-1)
print(reshape_arr)

#Flatten array
shape4_arr = np.array([[1, 2, 3], [4, 5, 6]])
flat_arr = shape4_arr.reshape(-1)
print(flat_arr)


#####
#Iterating Arrays
#####

it_arr = np.array([1, 2, 3])

for x in it_arr:
    print(x)


it2_arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in it2_arr:
    print(x)
    for y in x:
        print(y)


it3_arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in it3_arr:
    print(x)
    print('end of x')


for x in it3_arr:
  for y in x:
    for z in y:
      print(z)


#using nditer on array
it4_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(it4_arr):
    print(x)


#using op_dtype to change data type to string
it5_arr = np.array([1, 2, 3])

for x in np.nditer(it5_arr, flags=['buffered'], op_dtypes=['S']):
   print(x)


#iterate through ever scaler element for the 2D array skipping 1 element
it6_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(it6_arr[:, ::2]):
   print(x)

#use ndenumerate
it7_arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(it7_arr):
   print(idx, x)


it8_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(it8_arr):
   print(idx, x)



#Joining NumPy Arrays

#Join 2 arrays

j_arr1 = np.array([1, 2, 3])

j_arr2 = np.array([4, 5, 6]) 

#puts all items into the single array in the originating order based on the order of arrays
joined_arr = np.concatenate((j_arr2, j_arr1))

print(joined_arr)


#Join 2D arrays
j2_arr1 = np.array([[1, 2], [3, 4]])
j2_arr2 = np.array([[5, 6], [7, 8]])

joined2_arr = np.concatenate((j2_arr1, j2_arr2), axis=1)
print(joined2_arr)

#Stack arrays using first joined arrays
#creates a multi dimensional array, axis defines how many objects in each dimension
stacked_arr = np.stack((j_arr1, j_arr2), axis=1)
print(stacked_arr)

#Stack along rows (horizontal)
h_stack = np.hstack((j_arr1, j_arr2))
print(h_stack)

#stack along columns (vertical)
v_stack = np.vstack((j_arr1,j_arr2))
print(v_stack)

#stack along height(depth)
d_stack = np.dstack((j_arr1,j_arr2))
print(d_stack)



#Splitting Arrays

#split array into 3 parts
start_arr = np.array([1, 2, 3, 4, 5, 6])
split_arr = np.array_split(start_arr, 3)
print(split_arr)

#split into 4 arrays (the start_array has less elements than required)
#the last 2 of the new arrays only have one element
split2_arr = np.array_split(start_arr, 4)
print(split2_arr)

#Access the split arrays, using split_arr
print(split_arr[0])
print(split_arr[1])
print(split_arr[2])

#split 2D array into three 2D arrays
start2d_arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
split3_arr = np.array_split(start2d_arr, 3)
print(split3_arr)
print(split3_arr[0])
print(split3_arr[1])
print(split3_arr[2])

#split 2D array into three 2D arrays along rows
start2d1_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
split2drow_arr = np.array_split(start2d1_arr, 3, axis=1)
print(split2drow_arr)

#hsplit() 2d array into three 2d arrays along rows
hsplit_arr = np.hsplit(start2d1_arr, 3)
print(hsplit_arr)
print(hsplit_arr[0])
print(hsplit_arr[1])
print(hsplit_arr[2])



#Searching Arrays

#search with where()
search_arr = np.array([1, 2, 3, 4, 5, 4, 4])

#this gives the indexes where the value 4 is present
x = np.where(search_arr == 4)
print(x)


#Search for even values
search1_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

#this gives the indexes for even values
even = np.where(search1_arr%2 == 0)
print(even)


#Search for odd values in search1_arr
odd = np.where(search1_arr%2 ==1)
print(odd)


#searchsorted() searches an array to find where a value should be inserted
ss_arr = np.array([6, 7, 8, 9])

#going left to right, finds the index where 7 is not longer larger than the value
ss7 = np.searchsorted(ss_arr, 7)
print(ss7)

#search right to left
#going right to left, finds the index where 7 is no longer less than the next index
ss7right = np.searchsorted(ss_arr, 7, side='right')
print(ss7right)


#search multiple values
ss2_arr = np.array([1, 3, 5, 7])

#returns an array where the values in the search array would be inserted, left to right
#indexes in the serach and return arrays corrispond
sm = np.searchsorted(ss2_arr, [2,6,4])
print(sm)



#Sorting arrays

sort_arr = np.array([3, 2, 0, 1])

#Prints a copy of the array that has been sorted, original is unchanged
print(np.sort(sort_arr))
print(sort_arr)


#Sort strings
sortstring_arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(sortstring_arr))

#Sort boolean
#False, then True  assuming False = 0 and True = 1
sortbool_arr = np.array([True, False, True])
print(np.sort(sortbool_arr))


#Sort 2D arrays
sort2d_arr = np.array([[3, 2, 4], [5, 0, 1]])

#Sorts each array independently
print(np.sort(sort2d_arr))



#Filtering arrays

#create a boolean filter to use to build a new array 
#True brings the element to the new array

uf_arr = np.array([41, 42, 43, 44])

#array of boolean values
filter = [True, False, True, False]

filtered_arr = uf_arr[filter]
print(filtered_arr)


#retest the above with a larger array than filter
uf_arr = np.array([41, 42, 43, 44, 45, 46, 47, 48])

#array of boolean values
filter = [True, False, True, False]

#This doesn't work, the size of the source and filter arrays must match
filtered_arr = uf_arr[filter]
print(filtered_arr)


#Conditional Filter
#Create a filter that will puts elements with a value greater than 42 in a new array

#create empty array
new_filter = []

#for each loop with if statement to evaluate each element in the array 
for element in uf_arr:
   #if it is greater than 42, add to array
   if element > 42:
      new_filter.append(True)
   else:
      new_filter.append(False)

#apply created filter to source array
greaterthan42 = uf_arr[new_filter]

print(new_filter)
print(greaterthan42)


#Create a filter that will return even values from an array

uf2_arr = np.array([1, 2, 3, 4, 5, 6, 7])

even_filter = []

for element in uf2_arr:
   if element % 2 == 0:
      even_filter.append(True)
   else:
      even_filter.append(False)

even_arr = uf2_arr[even_filter]

print(even_filter)
print(even_arr)


#create a filter directly from an array

#filter array for elements greater than 42 in uf_arr
gt42_filter = uf_arr > 42

gt42_arr = uf_arr[gt42_filter]

print(gt42_filter)
print(gt42_arr)


#Retunr only elements with even values from uf2_arr

e_filter = uf2_arr % 2 == 0

e_arr = uf2_arr[e_filter]

print(e_filter)
print(e_arr)



#######
#NumPry Random
######

#need to run this, along with the original numpy if working in a new terminal
#import numpy as np
from numpy import random

#generate a random integer from 0 to 100
ri = random.randint(100)
print(ri)
#run the above multiple times to get different numbers

#generate a random float between 0 and 1
rf = random.rand()
print(rf)


#Generate a 1D array with 5 integers from 0 to 100
ri_arr = random.randint(100, size=(5))
print(ri_arr)


#Generate a 2-D array with 3 rows, each row with 5 random integers from 0 to 100
ri2d_ar = random.randint(100, size=(3, 5))
print(ri2d_ar)

#create 1-D array with 5 random floats
rf_arr = random.rand(5)
print(rf_arr)

#generate 2-D array with 3 rows, each with 5 random floats
rf2d_arr = random.rand(3, 5)
print(rf2d_arr)


#choice() method in random

#return a random value from a predefined array
rc = random.choice([3, 5, 7, 9])
print(rc)

#return random string
rs = random.choice(['f', 'v', '7', 'd'])
print(rs)


#use choice method with size parameter to return an array of values 
rc_arr = random.choice([3,5,7,9], size=(3,5))
print(rc_arr)


#Data Distribution
#create a 1-D array with a defined distibution of 100 elements
#distribution 3 = 10%, 5 = 30%, 7 = 60%, 9 = 0%
#in random.choice, the first array is the elements, the second array is probability linked by index (use p= to set the Probability parameter), then size of array
rcd_arr = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
print(rcd_arr)

#with the same distribution as above, create a 3 by 5 2-D array
rcd2d_arr = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3,5))
print(rcd2d_arr)


#Random Permitations of Elements

#Shuffle
test_arr = np.array([1, 2, 3, 4, 5])

#this will randomly re-order the elements and modify test_arr
random.shuffle(test_arr)
print(test_arr)


#Permutation
test2_arr = np.array([1,2,3,4,5])
#returns a re-ordered array, but does not modify the source array
print(random.permutation(test2_arr))
print(test2_arr)



########
#Seaborn Module 
#requires Matplotlib module
########

import matplotlib.pyplot as plt
import seaborn as sns


#plotting a Distplot
sns.distplot([0,1,2,3,4,5])
plt.show()

#plot without histogram
sns.distplot([0,1,2,3,4,5], hist=False)
plt.show()


#Normal Distribution

#generate a random normal distrubution size 2x3 (2 rows, 3 elements per row)
nd2x3 = random.normal(size=(2,3))
print(nd2x3)

#generate a random normal distribution with a mean at 1 and standard deviation at 2
#parameters loc=mean, scale=standard deviation, size=size of array
nd2x3m1sd2_arr = random.normal(loc=1, scale=2, size=(2,3))
print(nd2x3m1sd2_arr)

#visualization of Normal Distribution
sns.distplot(random.normal(size=1000), hist=False)
plt.show()


#Binomial Distibution

#binomial distributions are for the outcomes fo binary scenarios (flip of coin, positive/negative)
#random.binomial parameters n=number of trials, p=probability of occurence of each trial, size=shape of the returned array
#return 10 trials of flipping a coin for 10 data points
bd = random.binomial(n=10, p=.5, size=1000)
print(bd)
#This represents 100 flips of a coin, values in array the number of True(?) occurences

#visualize the above binomial distribution
sns.distplot(bd, hist=True, kde=False)
plt.show()


#normal distribution & random distribution comparison
#copied from: https://www.w3schools.com/python/numpy/numpy_random_binomial.asp 
sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')

plt.show()



#Poisson Distribution
#Estimation of how many times an even can occur in a specified time (If someone eats twice in a day, will they eat a third time)
#random.poisson parameters lam=rate or known number of occurences, size=shape of the returned array

#generate 1x10 poisson distribution for occurance of 2
pd_arr = random.poisson(lam=2, size=10)
print(pd_arr)

#visualiztion of above with 1000 elements
pd1000_arr = random.poisson(lam=2, size=1000)
sns.distplot(pd1000_arr, kde=False)
plt.show()


#Uniform Distribution https://www.w3schools.com/python/numpy/numpy_random_uniform.asp 


#Create a 2x3 uniform distribution sample
x = random.uniform(size=(2,3))
print(x)


#Visualize a random distribution
sns.distplot(random.uniform(size=1000), hist=False)
plt.show()


#Logistic Distribution
#Used to describe growth
#used extensively in logistic regression & neural networks

#create 2x3 array from a logistic distributin with a mean at 1 and standard deviation 2.0 
# loc = mean
# scale = standard deviation
# size = shape pf the array
x = random.logistic(loc=1, scale=2, size=(2,3))
print(x)

#visualise a logistic distribution
sns.distplot(random.logistic(size=1000), hist=False)
plt.show()

#visualize a normal & random distribution
sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(random.logistic(size=1000), hist=False, label='logistic')
plt.show()


#Multinomial Distribution
#show the distribution of rolling a six sided dice
# n = number of possible outcomes
# pvals = list of probabilities of the outcomes
#size = shape of the array
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x)

# Exponential Distribution
#descripts the until the next even (failure/success)
# scale = the inverse of rate
# size = shape of the array
x = random.exponential(scale=2, size=(2,3))
print(x)

#visualise exponential distribution
sns.distplot(random.exponential(size=1000), hist=False)
plt.show()

# https://www.w3schools.com/python/numpy/numpy_random_exponential.asp Quote from bottom the page:
#Relation Between Poisson and Exponential Distribution
#Poisson distribution deals with number of occurences of an event in a time period whereas exponential distribution deals with the time between these events.


# Chi Square Distribution
#Used as a basis to verify the hypothesis

# df = degree of freedom
# size = shape of the returned array
x = random.shisquare(df=2, size=(2,3))
print(x)

#visualize Chi Square
sns.distplot(random.chisquare(df=1, size=1000), hist=False)
plt.show()

#Skipping Rayeigh, Pareto & Zipf distributions






