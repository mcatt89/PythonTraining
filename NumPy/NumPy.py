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





