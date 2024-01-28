# Personal workbook by NumPy training from W3Schools
# https://www.w3schools.com/python/numpy/default.asp
# By Matt Herman
# Created January 28, 2024


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

