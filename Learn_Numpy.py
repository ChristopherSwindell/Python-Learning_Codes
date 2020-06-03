import numpy as np
import sys

##Create some lists
##a = np.array([1,2,3])
##print(a)
##b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
##print(b)
##Specify type of list to reduce storage size (e.g. if we have no big values)
##c = np.array([1,2,3], dtype = 'int16')


##Get Dimension
##print(a.ndim)
##print(b.ndim)

##Get Shape
##print(a.shape)
##print(b.shape)
##Note:
##    a has 1 row, 3 columns
##    b has 2 rows, 3 columns

##Get Type
##print(a.dtype)
##print(b.dtype)
##print(c.dtype)

##Get Size
##print(a.itemsize)
##print(b.itemsize)
##print(c.itemsize)

##Total Size = size * itemsize
##print(a.size)
##print(a.itemsize)
##print(a.size * a.itemsize)
##print(a.nbytes)

'''Accessing/Changing specific elements, rows, columns, etc.'''
##a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
##print(a)
##print(a.shape)

##Get a specific element [r, c]
##print(a[1,5])
##print(a[1,-2])

##Get a specific row
##print(a[0,:])

##Get a specific column
##print(a[:,2])

##Getting a little more fancy [starindex:endindex:stepsize]
## Start at 2, go through 6, and step through by 2
##print(a[0, 1:6:2])

##Change element
##a[1,5] = 20
##print(a)

##Change column 3 to all fives
##a[:,2] = 5
##print(a)

##Change column 3 to 1 and 2
##a[:,2] = [1,2]
##print(a)

## 3-d example
##b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
##print(b)

##Get specific element (work outside in)
##print(b[0,1,1])

##Replace
##b[:,1,:] = [[9,9],[8,8]]
##print(b)

'''Initializing Different Types of Arrays'''
##All 0s matrix
##a = np.zeros(5)
##print("1X5 matrix of 0s is",a)
##b = np.zeros((2,3,3))
##print("2X3X3 matrix of 0s is",b)
##print(b)

##All 1s matrix
##c = np.ones((4,2,2), dtype='int32')
##print("4X2X2 matrix of 1s",c)

##Any other number
##d = np.full((2,2),99)
##print(d)

##Any other number (full_like)
##e = np.full_like(a, 4)

## Random decimal numbers
##f = np.random.rand(4,2)
##print(f)
##g = np.random.random_sample(a.shape)
##print(g)
##h = np.random.randint(7, size = (3,3))
##print(h)
##i = np.random.randint(-4,8,size=(3,3))
##print("selects random numbers -4 through 7",i)

##Identity matrix
##j = np.identity(5)
##print(j)

##repeat array
##arr = np.array([1,2,3])
##r1 = np.repeat(arr,3)
##print(r1)
##arr2 = np.array([[1,2,3]])
##r2 = np.repeat(arr2, 3, axis = 0)
##print(r2)

##Building a matrix
##arr1 = np.ones((5,5), dtype='int32')
##arr1[1:4,1:4]=0
##arr1[2,2]=9
##print(arr1)
##print(arr2)

##Be careful when copying arrays to use .copy()
##a = np.array([1,2,3])
##b = a.copy()

'''Mathematics'''
##a = np.array([1,2,3,4])
##print(a)
##print(a+2)
##print(a-2)
##print(a*2)
##print(a/2)
##print(a**2)

##b = np.array([1,0,1,0])
##print(a+b)

##print(np.sin(a))

'''Linear Algebra'''
##a = np.ones((2,3),'int32')
##b = np.full((3,2),2)
##print(a)
##print(b)

##Multiply matices
##c = np.matmul(a,b)
##print(c)

##Find determinant
##d = np.identity(3)
##d1 = np.linalg.det(d)
##print(d1)

'''Statistics'''
##stats = np.array(([1,2,3],[4,5,6]))
##print(stats)
##mini = np.min(stats)
##print(mini)
##maxi = np.max(stats)
##print(maxi)
##mini1 = np.min(stats, axis = 0)
##print(mini1)
##maxi1 = np.max(stats, axis = 1)
##print(maxi1)
##sum1 = np.sum(stats)
##print(sum1)

'''Reorganizing Arrays'''
##before = np.array([[1,2,3,4],[5,6,7,8]])
##print(before)
##after = before.reshape((8,1))
##print(after)

'''Vertically stacking vectors'''
##v1 = np.array([1,2,3,4])
##v2 = np.array([5,6,7,8])
##v12 = np.vstack([v1,v2])
##print(v12)

'''Horizontally stacking vectors'''
##h1 = np.ones((2,4))
##h2 = np.zeros((2,2))
##h12 = np.hstack((h1,h2))
##print(h12)

'''Miscellaneous'''
##Load Data from File
##filedata = np.genfromtxt('data.txt', delimiter = ',')
##filedata = filedata.astype('int32')

##Boolean Masking and Advanced Indexing
##filedata > 50
##filedata[filedata > 50]
##np.any(filedata > 50, axis = 0)
##np.all(filedata > 50, axis = 1)
##((filedata > 50) & (filedata < 100))
##(~(filedata > 50) & (filedata < 100))

##You can index with a list
##a = np.array([1,2,3,4,5,6,7,8,9])
##a[[1,2,8]]
##print(a)

arr1 = np.array([1,2,3,4,5])
##print(arr1)
arr2 = np.full((1,5),5)
##print(arr2)
arr3 = np.vstack((arr1,arr2+arr1,2*arr2+arr1,3*arr2+arr1,4*arr2+arr1,5*arr2+arr1))
##print(arr3)
index1 = arr3[2:4,0:2]
##print(index1)
index2 = arr3[[0,1,2,3],[1,2,3,4]]
##print(index2)
index3 = arr3[[0,4,5],3:]
print(index3)
