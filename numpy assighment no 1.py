#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
x = np.float32(1.0)
x

y = np.int_([1,2,4])
y

z = np.arange(3, dtype=np.uint8)
z


# In[8]:


z.astype(float)                 

np.int8(z)


# In[4]:


np.array([1, 2, 3], dtype='f')


# In[9]:


z.dtype


# In[10]:


d = np.dtype(int)
d


np.issubdtype(d, np.integer)


np.issubdtype(d, np.floating)


# # Overflow Errors

# In[11]:


np.power(100, 8, dtype=np.int64)

np.power(100, 8, dtype=np.int32)


# In[12]:


np.iinfo(int) # Bounds of the default integer on this system.

np.iinfo(np.int32) # Bounds of a 32-bit integer

np.iinfo(np.int64) # Bounds of a 64-bit integer


# # Converting Python array_like Objects to NumPy Arrays

# In[14]:


x = np.array([2,3,1,0])
x = np.array([2, 3, 1, 0])
x = np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists,

x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])


# # Intrinsic NumPy Array Creation¶

# In[16]:


np.zeros((2, 3))


# In[17]:


np.arange(10)

np.arange(2, 10, dtype=float)

np.arange(2, 3, 0.1)


# In[18]:


np.linspace(1., 4., 6)


# In[19]:


np.indices((3,3))


# # The delimiter argument

# In[ ]:


data = u"1, 2, 3\n4, 5, 6"
np.genfromtxt(StringIO(data), delimiter=",")


# In[ ]:


data = u"  1  2  3\n  4  5 67\n890123  4"
np.genfromtxt(StringIO(data), delimiter=3)



data = u"123456789\n   4  7 9\n   4567 9"
np.genfromtxt(StringIO(data), delimiter=(4, 3, 2))


# # The autostrip argument

# In[ ]:


data = u"""#
# Skip me !
# Skip me too !
1, 2
3, 4
5, 6 #This is the third line of the data
7, 8
# And here comes the last line
9, 0
"""
np.genfromtxt(StringIO(data), comments="#", delimiter=",")


# # Skipping lines and choosing columns

# In[ ]:


data = u"\n".join(str(i) for i in range(10))
np.genfromtxt(StringIO(data),)

np.genfromtxt(StringIO(data),
              skip_header=3, skip_footer=5)


# # The usecols argument

# In[ ]:


data = u"1 2 3\n4 5 6"
np.genfromtxt(StringIO(data), usecols=(0, -1))


# In[ ]:


data = u"1 2 3\n4 5 6"
np.genfromtxt(StringIO(data),
              names="a, b, c", usecols=("a", "c"))


np.genfromtxt(StringIO(data),
              names="a, b, c", usecols=("a, c"))


# # The names argument

# In[ ]:


data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, dtype=[(_, int) for _ in "abc"])


# In[ ]:



 data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, names="A, B, C")


# # The defaultfmt argument

# In[ ]:


data = StringIO("1 2 3\n 4 5 6")
 np.genfromtxt(data, dtype=(int, float, int))
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '<i8')])


# In[ ]:


data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, dtype=(int, float, int))


# In[ ]:


data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, dtype=(int, float, int), names="a")


#We can overwrite this default with the defaultfmt argument, that takes any format string:


 data = StringIO("1 2 3\n 4 5 6") np.genfromtxt(data, dtype=(int, float, int), defaultfmt="var_%02i")
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('var_00', '<i8'), ('var_01', '<f8'), ('var_02', '<i8')])


# # Single element indexing¶

# In[37]:


x = np.arange(10)
x[2]
2
x[-2]
8


# In[38]:


x = np.arange(10)
x[2]

x[-2]


# In[39]:


x.shape = (2,5) # now x is 2-dimensional
x[1,3]
8
x[1,-1]
9


# # Other indexing options

# In[41]:


x = np.arange(10)
x[2:5]

x[:-7]

x[1:7:2]

y = np.arange(35).reshape(5,7)
y[1:5:2,::3]


# # Index arrays

# In[43]:


x = np.arange(10,1,-1)
x

x[np.array([3, 3, 1, 8])]


# # Indexing Multi-dimensional arrays

# In[44]:


y[np.array([0,2,4]), np.array([0,1,2])]


# In[45]:


y[np.array([0,2,4])]


# # Boolean or “mask” index arrays

# In[46]:


b = y>20
y[b]


# In[47]:


x = np.arange(30).reshape(2,3,5)
x






b = np.array([[True, True, False], [False, True, True]])
x[b]


# # Combining index arrays with slices

# In[48]:


y[np.array([0, 2, 4]), 1:3]


# # Structural indexing tools

# In[49]:


y.shape

y[:,np.newaxis,:].shape


# In[50]:


x = np.arange(5)
x[:,np.newaxis] + x[np.newaxis,:]


# # Dealing with variable numbers of indices within programs

# In[ ]:


indices = (1, Ellipsis, 1) # same as [1,...,1]
z[indices]


# # General Broadcasting Rules

# In[ ]:


Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3


# # Byte-swapping

# In[ ]:


import numpy as np
big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_buffer)
big_end_arr[0]

big_end_arr[1]


# # Data and dtype endianness don’t match, change dtype to match data

# In[ ]:


wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_buffer)
wrong_end_dtype_arr[0]


# # Data and dtype endianness match, swap data and dtype

# In[ ]:


swapped_end_arr = big_end_arr.byteswap().newbyteorder()
swapped_end_arr[0]

swapped_end_arr.tobytes() == big_end_buffer


# In[ ]:


swapped_end_arr = big_end_arr.byteswap().newbyteorder()
swapped_end_arr[0]

swapped_end_arr.tobytes() == big_end_buffer


# # Structured arrays

# In[59]:


x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
             dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
x


# # Structured Datatype Creation

# In[ ]:


np.dtype([('x', 'f4'), ('y', np.float32), ('z', 'f4', (2, 2))])
dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4', (2, 2))])


# # Manipulating and Displaying Structured Datatypes

# In[61]:


d = np.dtype([('x', 'i8'), ('y', 'f4')])
d.names
('x', 'y')


# # Automatic Byte Offsets and Alignment

# In[62]:


def print_offsets(d):
    print("offsets:", [d.fields[name][1] for name in d.names])
    print("itemsize:", d.itemsize)
print_offsets(np.dtype('u1, u1, i4, u1, i8, u2'))


# # Field Titles¶

# In[63]:


np.dtype([(('my title', 'name'), 'f4')])


# # Assigning data to a Structured Array

# In[64]:


x = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
x[1] = (7, 8, 9)
x


# # Indexing Structured Arrays

# In[65]:


x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
x['foo']

x['foo'] = 10
x


# # Structure Comparison

# In[66]:


a = np.zeros(2, dtype=[('a', 'i4'), ('b', 'i4')])
b = np.ones(2, dtype=[('a', 'i4'), ('b', 'i4')])
a == b


# # Record Arrays

# In[67]:


recordarr = np.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
                   dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
recordarr.bar

recordarr[1:2]


recordarr[1:2].foo

recordarr.foo[1:2]

recordarr[1].baz


# # View casting

# In[68]:


import numpy as np
# create a completely useless ndarray subclass
class C(np.ndarray): pass
# create a standard ndarray
arr = np.zeros((3,))
# take a view of it, as our useless subclass
c_arr = arr.view(C)
type(c_arr)


# # Creating new from template

# In[69]:


v = c_arr[1:]
type(v) # the view is of type 'C'

v is c_arr # but it's a new instanc


# # Explicit constructor

# In[71]:


# Explicit constructor
c = C((10,))





# View casting
a = np.arange(10)
cast_a = a.view(C)



# Slicing (example of new-from-template)
cv = c[:1]


# # Simple example - adding an extra attribute to ndarray

# In[72]:


import numpy as np

class InfoArray(np.ndarray):

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(InfoArray, subtype).__new__(subtype, shape, dtype,
                                                buffer, offset, strides,
                                                order)
        # set the new 'info' attribute to the value passed
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.info = getattr(obj, 'info', None)
        # We do not need to return anything


# In[73]:


obj = InfoArray(shape=(3,)) # explicit constructor
type(obj)

obj.info is None

obj = InfoArray(shape=(3,), info='information')
obj.info

v = obj[1:] # new-from-template - here - slicing
type(v)

v.info

arr = np.arange(10)
cast_arr = arr.view(InfoArray) # view casting
type(cast_arr)

cast_arr.info is None


# # Slightly more realistic example - attribute added to existing array

# In[74]:


import numpy as np

class RealisticInfoArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)


# In[75]:


arr = np.arange(5)
obj = RealisticInfoArray(arr, info='information')
type(obj)

obj.info

v = obj[1:]
type(v)

v.info


# # __array_ufunc__ for ufuncs

# In[ ]:


input numpy as np

class A(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, A):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, A):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super(A, self).__array_ufunc__(ufunc, method,
                                                 *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], A):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(A)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], A):
            results[0].info = info

        return results[0] if len(results) == 1 else results


# In[ ]:


a = np.arange(5.).view(A)
b = np.sin(a)
b.info

b = np.sin(np.arange(5.), out=(a,))
b.info

a = np.arange(5.).view(A)
b = np.ones(1).view(A)
c = a + b
c.info

a += b
a.info


# # Extra gotchas - custom __del__ methods and ndarray.base

# In[78]:


# A normal ndarray, that owns its own data
arr = np.zeros((4,))
# In this case, base is None
arr.base is None

# We take a view
v1 = arr[1:]
# base now points to the array that it derived from
v1.base is arr

# Take a view of a view
v2 = v1[1:]
# base points to the view it derived from
v2.base is v1


# # Subclassing and Downstream Compatibility

# In[ ]:


def sum(self, axis=None, dtype=None, out=None, keepdims=False):


# In[ ]:




