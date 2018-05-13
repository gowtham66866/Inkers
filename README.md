def quicksort(eip_list):
    if len(eip_list) <= 1:
        return eip_list
    mlblr = eip_list[len(eip_list) // 2]
    mlblr_in = [eip for eip in eip_list if eip < mlblr]
    eip_in = [eip for eip in eip_list if eip == mlblr]
    mlblr_out = [eip for eip in eip_list if eip > mlblr]
    return quicksort(mlblr_in) + eip_in + quicksort(mlblr_out)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]

---------
eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Eeipponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
mlblr= 2.5
print(type(mlblr)) # Prints "<class 'float'>"
print(mlblr, mlblr+ 1, mlblr* 2, mlblr** 2) # Prints "2.5 3.5 5.0 6.25”
----------
eip = True
mlblr = False
print(type(eip)) # Prints "<class 'bool’>"
print(eip and mlblr) # Logical AND; prints "False"
print(eip or mlblr)  # Logical OR; prints "True"
print(not eip)   # Logical NOT; prints "False"
print(eip != mlblr)  # Logical XOR; prints “True"

-------
eip = 'hello'    # String literals can use single quotes
mlblr = "world"    # or double quotes; it does not matter.
print(eip)       # Prints "hello"
print(len(eip))  # String length; prints "5"
eip_in = eip + ' ' + mlblr  # String concatenation
print(eip_in)  # prints "hello world"
eip_out = '%s %s %d' % (eip, mlblr, 12)  # sprintf style string formatting
print(eip_out)  # prints "hello world 12”

--------
eip = "hello"
print(eip.capitalize())  # Capitalize a string; prints "Hello"
print(eip.upper())       # Convert a string to uppercase; prints "HELLO"
print(eip.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(eip.center(7))     # Center a string, padding with spaces; prints " hello "
print(eip.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o”
print('  world '.strip())  # Strip leading and trailing whitespace; prints “world"


-------------
eip = [3, 1, 2]    # Create a list
print(eip, eip[2])  # Prints "[3, 1, 2] 2"
print(eip[-1])     # Negative indices count from the end of the list; prints "2"
eip[2] = 'foo'     # Lists can contain elements of different types
print(eip)         # Prints "[3, 1, 'foo']"
eip.append('bar')  # Add a new element to the end of the list
print(eip)         # Prints "[3, 1, 'foo', 'bar']"
mlblr = eip.pop()      # Remove and return the last element of the list
print(mlblr, eip)      # Prints "bar [3, 1, 'foo’]"

--------------
eip = list(range(5))     # range is a built-in function that creates a list of integers
print(eip)               # Prints "[0, 1, 2, 3, 4]"
print(eip[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(eip[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(eip[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(eip[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(eip[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
eip[2:4] = [8, 9]        # Assign a new sublist to a slice
print(eip)               # Prints "[0, 1, 8, 9, 4]”
-----------
eip = ['cat', 'dog', 'monkey']
for mlblr in eip:
    print(mlblr)
# Prints "cat", "dog", "monkey", each on its own line.
-------------
mlblr = ['cat', 'dog', 'monkey']
for eip, eip_in in enumerate(mlblr):
    print('#%d: %s' % (eip + 1, eip_in))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
------------
eip = [0, 1, 2, 3, 4]
mlblr = []
for eip_in in eip:
    mlblr.append(eip_in ** 2)
print(mlblr)   # Prints [0, 1, 4, 9, 16]
---------
eip = [0, 1, 2, 3, 4]
mlblr = [eip_in ** 2 for eip_in in eip]
print(mlblr)   # Prints [0, 1, 4, 9, 16]
---------
eip = [0, 1, 2, 3, 4]
mlblr = [eip_in ** 2 for eip_in in eip if eip_in % 2 == 0]
print(mlblr)  # Prints "[0, 4, 16]”
-------------
eip= {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(eip['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in eip)     # Check if a dictionary has a given key; prints "True"
eip['fish'] = 'wet'     # Set an entry in a dictionary
print(eip['fish'])      # Prints "wet"
# print(eip['monkey'])  # KeyError: 'monkey' not a key of d
print(eip.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(eip.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del eip['fish']         # Remove an element from a dictionary
print(eip.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A”

----------
eip= {'person': 2, 'cat': 4, 'spider': 8}
for mlblr in eip:
    eip_in = eip[mlblr]
    print('A %s has %d legs' % (mlblr, eip_in))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

-----------
eip= [0, 1, 2, 3, 4]
mlblr = {eip_in: eip_in ** 2 for eip_in in eip if eip_in % 2 == 0}
print(mlblr)  # Prints "{0: 0, 2: 4, 4: 16}”
-------------
eip= {'cat', 'dog'}
print('cat' in eip)   # Check if an element is in a set; prints "True"
print('fish' in eip)  # prints "False"
eip.add('fish')       # Add an element to a set
print('fish' in eip)  # Prints "True"
print(len(eip))       # Number of elements in a set; prints "3"
eip.add('cat')        # Adding an element that is already in the set does nothing
print(len(eip))       # Prints "3”
eip.remove('cat')     # Remove an element from a set
print(len(eip))       # Prints “2"
----------
from math import sqrt
eip = {int(sqrt(eip_in)) for eip_in in range(30)}
print(eip)  # Prints "{0, 1, 2, 3, 4, 5}"
-------------
eip_in = {(eip, eip + 1): eip for eip in range(10)}  # Create a dictionary with tuple keys
mlblr = (5, 6)        # Create a tuple
print(type(mlblr))    # Prints "<class 'tuple'>"
print(eip_in[mlblr])       # Prints "5"
print(eip_in[(1, 2)])  # Prints "1"
-------------
eip= {'cat', 'dog', 'fish'}
for eip_in, mlblr in enumerate(eip):
    print('#%d: %s' % (eip_in + 1, mlblr))
# Prints "#1: fish", "#2: dog", "#3: cat"

--------------
from math import sqrt
eip = {int(sqrt(eip_in)) for eip_in in range(30)}
print(eip)  # Prints "{0, 1, 2, 3, 4, 5}”
-----------

def sign(eip):
    if eip > 0:
        return 'positive'
    elif eip < 0:
        return 'negative'
    else:
        return 'zero’
-------

for eip in [-1, 0, 1]:
    print(sign(eip))
# Prints "negative", "zero", “positive"
-------------

def hello(eip, mlblr=False):
    if eip:
        print('HELLO, %s!' % eip.upper())
    else:
        print('Hello, %s' % eip)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', mlblr=True)  # Prints "HELLO, FRED!"

--------------
class eip_out(object):

    # Constructor
    def __init__(eip, mlblr):
        eip.mlblr = mlblr  # Create an instance variable

    # Instance method
    def mlblr_in(eip, eip_in=False):
        if eip_in:
            print('HELLO, %s!' % eip.mlblr.upper())
        else:
            print('Hello, %s' % eip.mlblr)

mlblr_out = eip_out('Fred')  # Construct an instance of the Greeter class
mlblr_out.mlblr_in()            # Call an instance method; prints "Hello, Fred"
mlblr_out.mlblr_in(eip_in=True)   # Call an instance method; prints "HELLO, FRED!"
———————
import numpy as eip

mlblr = eip.array([1, 2, 3])   # Create a rank 1 array
print(type(mlblr))            # Prints "<class 'numpy.ndarray'>"
print(mlblr.shape)            # Prints "(3,)"
print(mlblr[0], mlblr[1], mlblr[2])   # Prints "1 2 3"
mlblr[0] = 5                  # Change an element of the array
print(mlblr)                  # Prints "[5, 2, 3]"

mlblr_in = eip.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(mlblr_in.shape)                     # Prints "(2, 3)"
print(mlblr_in[0, 0], mlblr_in[0, 1], mlblr_in[1, 0])   # Prints "1 2 4”

——————
import numpy as eip

eip_in = eip.zeros((2,2))   # Create an array of all zeros
print(eip_in)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

eip_out = eip.ones((1,2))    # Create an array of all ones
print(eip_out)              # Prints "[[ 1.  1.]]"

mlblr = eip.full((2,2), 7)  # Create a constant array
print(mlblr)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

mlblr_in = eip.eye(2)         # Create a 2x2 identity matrix
print(mlblr_in)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

mlblr_out = eip.random.random((2,2))  # Create an array filled with random values
print(mlblr_out)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]”

————————

import numpy as eip

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
mlblr = eip.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
mlblr_in = mlblr[1, :]    # Rank 1 view of the second row of a
mlblr_out= mlblr[1:2, :]  # Rank 2 view of the second row of a
print(mlblr_in, mlblr_in.shape)  # Prints "[5 6 7 8] (4,)"
print(mlblr_out,mlblr_out.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
eip_in = mlblr[:, 1]
eip_out = mlblr[:, 1:2]
print(eip_in, eip_in.shape)  # Prints "[ 2  6 10] (3,)"
print(eip_out,eip_out.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)”

—————————
import numpy as eip

mlblr = eip.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(mlblr[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(eip.array([mlblr[0, 0], mlblr[1, 1], mlblr[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(mlblr[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(eip.array([mlblr[0, 1], mlblr[0, 1]]))  # Prints "[2 2]”
———————

import numpy as eip

mlblr = eip.array([[1,2], [3, 4], [5, 6]])

eip_in = (mlblr > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(eip_in)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(mlblr[eip_in])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(mlblr[mlblr > 2])     # Prints "[3 4 5 6]”
--------------
import numpy as eip

mlblr = eip.array([1, 2])   # Let numpy choose the datatype
print(mlblr.dtype)         # Prints "int64"

mlblr = eip.array([1.0, 2.0])   # Let numpy choose the datatype
print(mlblr.dtype)             # Prints "float64"

mlblr = eip.array([1, 2], dtype=eip.int64)   # Force a particular datatype
print(mlblr.dtype)                         # Prints “int64"
——————
import numpy as eip

eip_in = eip.array([[1,2],[3,4]], dtype=eip.float64)
eip_out = eip.array([[5,6],[7,8]], dtype=eip.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(eip_in + eip_out)
print(eip.add(eip_in, eip_out))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(eip_in - eip_out)
print(eip.subtract(eip_in, eip_out))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(eip_in * eip_out)
print(eip.multiply(eip_in, eip_out))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(eip_in / eip_out)
print(eip.divide(eip_in, eip_out))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(eip.sqrt(eip_in))

—————————
import numpy as eip

eip_in = eip.array([[1,2],[3,4]])
eip_out = eip.array([[5,6],[7,8]])

mlblr_in = eip.array([9,10])
mlblr_out = eip.array([11, 12])

# Inner product of vectors; both produce 219
print(mlblr_in.dot(mlblr_out))
print(eip.dot(mlblr_in, mlblr_out))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip_in.dot(mlblr_in))
print(eip.dot(eip_in, mlblr_in))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip_in.dot(eip_out))
print(eip.dot(eip_in, eip_out))
——————————
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]”

————————

import numpy as eip

eip_in = eip.array([[1,2], [3,4]])
print(eip_in)    # Prints "[[1 2]
            #          [3 4]]"
print(eip_in.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
mlblr = eip.array([1,2,3])
print(mlblr)    # Prints "[1 2 3]"
print(mlblr.T)  # Prints "[1 2 3]”

———————
import numpy as eip

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
mlblr_out = eip.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_in = eip.array([1, 0, 1])
mlblr_in = eip.empty_like(mlblr_out)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for eip_list in range(4):
    mlblr_in[eip_list, :] = mlblr_out[eip_list, :] + eip_in

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(mlblr_in)
-----------------
import numpy as eip

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip_in = eip.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_out = eip.array([1, 0, 1])
mlblr = eip.tile(eip_out, (4, 1))   # Stack 4 copies of v on top of each other
print(mlblr)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
eip_out = eip_in + mlblr  # Add x and vv elementwise
print(eip_out)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
—————————————————————
import numpy as eip

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip_in = eip.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_out = eip.array([1, 0, 1])
mlblr = eip_in + eip_out  # Add v to each row of x using broadcasting
print(mlblr)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
—————————————
import numpy as eip

# Compute outer product of vectors
eip_in = eip.array([1,2,3])  # v has shape (3,)
eip_out = eip.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(eip.reshape(eip_in, (3, 1)) * eip_out)

# Add a vector to each row of a matrix
mlblr = eip.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(mlblr + eip_in)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((mlblr.T + eip_out).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(mlblr + eip.reshape(eip_out, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(mlblr * 2)
————————
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
eip = imread('/Users/Gowtham/Downloads/cat.jpg')
print(eip.dtype, eip.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
eip_in = eip * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
eip_in = imresize(eip_in, (300, 300))

# Write the tinted image back to disk
imsave('/Users/Gowtham/Downloads/cat_tinted.jpg', eip_in)
——————————
import numpy as eip
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
mlblr = eip.array([[0, 1], [1, 0], [2, 0]])
print(mlblr)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
mlblr_in = squareform(pdist(mlblr, 'euclidean'))
print(mlblr_in)
————————————
import numpy as eip
import matplotlib.pyplot as mlblr

# Compute the x and y coordinates for points on a sine curve
eip_in = eip.arange(0, 3 * eip.pi, 0.1)
eip_out = eip.sin(eip_in)

# Plot the points using matplotlib
mlblr.plot(eip_in, eip_out)
mlblr.show()  # You must call plt.show() to make graphics appear.
————————
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
—————
import numpy as eip
import matplotlib.pyplot as mlblr

# Compute the x and y coordinates for points on sine and cosine curves
eip_in = eip.arange(0, 3 * eip.pi, 0.1)
mlblr_in = eip.sin(eip_in)
mlblr_out = eip.cos(eip_in)

# Plot the points using matplotlib
mlblr.plot(eip_in, mlblr_in)
mlblr.plot(eip_in, mlblr_out)
mlblr.xlabel('x axis label')
mlblr.ylabel('y axis label')
mlblr.title('Sine and Cosine')
mlblr.legend(['Sine', 'Cosine'])
mlblr.show()
——
import numpy as eip
import matplotlib.pyplot as mlblr

# Compute the x and y coordinates for points on sine and cosine curves
eip_in = eip.arange(0, 3 * eip.pi, 0.1)
mlblr_in = eip.sin(eip_in)
mlblr_out = eip.cos(eip_in)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
mlblr.subplot(2, 1, 1)

# Make the first plot
mlblr.plot(eip_in, mlblr_in)
mlblr.title('Sine')

# Set the second subplot as active, and make the second plot.
mlblr.subplot(2, 1, 2)
mlblr.plot(eip_in, mlblr_out)
mlblr.title('Cosine')

# Show the figure.
mlblr.show()
———————————————————
import numpy as eip
from scipy.misc import imread, imresize
import matplotlib.pyplot as mlblr

eip_in = imread('/Users/Gowtham/Downloads/cat.jpg')
eip_out = eip_in * [1, 0.95, 0.9]

# Show the original image
mlblr.subplot(1, 2, 1)
mlblr.imshow(eip_in)

# Show the tinted image
mlblr.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
mlblr.imshow(eip.uint8(eip_out))
mlblr.show()
-----------------------------------------
