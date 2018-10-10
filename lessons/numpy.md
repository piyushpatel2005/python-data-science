# NumPy Library

NumPy support greater variety of numerical types than Python does. It provides user with multidimensional arrays, along with a large set of functions to operate on these arrays.

Numpy can be installed using `pip` as follows:

`pip install numpy`

A NumPy array would require much less memory to store the same amount of data compared to a Python list which makes reading and writing from the array faster.

Check the [interactive notebook for NumPy](numpy.ipynb)

[Numpy](numpy2.ipynb)

```python
import numpy as np

# Create array
mylist = [1, 2, 3]
x = np.array(mylist)

# another way
y = np.array([1, 2, 3])

# create multidimensional array using list of lists
m = np.array([[1, 2, 3], [4, 5, 6]])  # [[1, 2, 3],
                                      #  [4, 5, 6]]

# find dimension of array using shape method
m.shape     # (2, 3)

n = np.arange(0, 5, 1)    # [0, 1, 2, 3, 4]

# in reshape the size of the array must not change. total elements must be the same
n = n.reshape(3, 5)     # reshape array to be 3 x 5

o = np.linspace(0, 4, 9)    # creates 9 evenly spaced elements from 0 to 4
# Prints array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.])

o.resize(3,3)

np.ones((3,2))  # returns 3 x 2 matrix with all element as 1.

np.zeros((2,3))  # returns 2 x 3 array with all 0s.

np.eye(3)   # creates 3 x 3 unit matrix, diagonal as 1

np.diag(n)   # extracts a diagonal or constructs a diagonal array depending on input array n.
```

There are so many array manipulation methods in Numpy.

```python
np.array([1,2,3] * 3)   # array([1,2,3,1,2,3,1,2,3])

np.repeat([1,2,3], 3)   # [1,1,1,2,2,2,3,3,3]

# Combine arrays
p = np.ones([2,3], int)   # p = [[1, 1, 1],
                            #    [1, 1, 1]]

# stack the array vertically
np.vstack([p, 2*p])       # [[1, 1, 1],
                          # [1, 1, 1],
                          # [2, 2, 2],
                          # [2, 2, 2]]

np.hstack(p, 2*p)           # [1, 1, 1, 2, 2, 2],
                            # [1, 1, 1, 2, 2, 2]]

```

Numpy supports various arithmetic operations on arrays directly.

```python
print (x + y)   # elementwise addition
print (x - y)   # elementwise subtraction

print(x * y)    # elementwise multiplication
print (x / y)   # elementwise division

print(x ** 2)   # elementwise power

# dot product
x.dot(y)

print(len(x))   # number of rows of array x

z = [[4, 5, 6], [16, 25, 36]]
z.shape   # (2,3)
# Transpose means row number and column numbers are interchanged. 2x3 array becomes 3x2.

z.T   # [[4,16], [5, 25], [6, 36]]
z.shape   # (3, 2)

z.dtype   # to see the data type of the elements in the array

# cast elements to specific type
z = z.astype('f')
z.dtype   # dtype('float32')
```

```python
a = np.array([-4, -2, 1, 3, 5])
a.sum()   # sum of all elements
a.max()   # maximum element
a.min()   # minimum element
a.mean()  # mean of all elements
a.std()   # standard deviation
a.argmax()    # return the index of maximum element, here 4
a.argmin()    # 0
```

```python
s = np.arange(5) ** 2   # [0, 1, 4, 9, 16]
s[1:4]    # [1, 4, 9]
# slicing works the same and indexing also works like lists.

array[row, column]    # would give row number and column number element

r[3, 3:6]   # gives 3rd row from 3 to 6 elements in row array
r[:2, :-1]  # gives first two rows and in each row every element except last one.

r[r>30]   # will return all elements greater than 30
```

Copy Data

```python
r2 = r[:3, :3]
r2[:] = 0   # this will change values for r2 as well as r

# To avoid this we can get a copy of array using r.copy()
r_copy = r.copy()   # now manipulate r_copy, r will not change.

# create array with random elements
test = np.random.randint(0, 10, (4,3))    # create 4x3 array with random elements from 0 to 10

for row in test:
  print(row)

for i in range(len(test)):
  print(test[i])

for i, row in enumerate(test):
  print('row', i, 'is', row)

test2 = test ** 2   # change all elements to square value

# We can use zip to iterate through two arrays
for i, j in zip(test, test2):
  print(i, '+', j, '= ', i+j)   # add each element.
```
