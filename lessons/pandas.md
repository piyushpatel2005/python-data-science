# Pandas Library for Data Science

Pandads library provides many common functionality for Data science and makes accessing and manipulating and combinting data too easy. Pandas deals with everything that Numpy and SciPy cannot do. It has object data structures, DataFrames and Series. It allows to handle complex tables of data of different types and time series.

To install Pandas:

`pip install pandas`

Pandas library brings richness of R in the world of Python to handle data. It has three data structures.

1. Series
2. DataFrame
3. Panel

Checkout [interactive session with Pandas](pandas.ipynb)

[Pandas interactive session 2](pandas2.ipynb)

## Series

It comes with index and data. Data column has column name so it can be accessed using dot operator.

```python
import pandas as import pd
pd.Series?    # shows documentation

animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

numbers = [1, 2, 3]
pd.Series(numbers)
```

Pandas identify the type of data passed in and convert them to appropriate type for efficiency.

In Python we have `None` type, but when using numbers, Pandas uses `NaN` which is special floating point number.

```python
animals = ['Tiger', 'Bear', None]
pd.Series(animals)  # puts None in string data type.

numbers = [1, 2, None]
pd.Series (numbers)   # places NaN instead of None

import numpy as np
np.nan == None  # False

np.nan == np.nan    # False

# use isnan() function to check whether value is NaN
np.isnan(np.nan)    # True

sports = {'Archery': 'Bhutan',
        'Gold': 'Scotland'}
s = pd.Series(sports)
s
# Archery         Bhutan
# Golf            Scotland

s.index   # Index(['Archery', 'Golf'])

s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
# this creates first list as data and second list as index.

# If we want list of only few elements.
sports = {
  'Archery': 'Bhutan',
  'Cricket': 'India',
  'Golf': 'Scotland',
  'Sumo': 'Japan'
}
s = pd.Series(sports, index=['Cricket', 'Golf'])    # would return only two elements in series.
```

```python
sports = {'Achery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'
          }
pd.Series(sports)
s

s.iloc[3]   # South Korea, get value by index location.

s.loc['Golf']   # Scotland

s[3]    # South Korea, not advisable in Pandas, use iloc attribute

s['Golf']   # Scotland, not advisable in Pandas, use loc attribute
```

## Mathematics

```python
import numpy as np

s = pd.Series([100.00, 120.0, 101.0, 3.0])

total = np.num(s)   # faster
print(total)

s = pd.Series(np.random.randint(0, 1000, 10000))
s.head()    # returns first five elements of series.

len(s)    # 10000
```

We can see that numpy works efficiently if we compare function with Python. Pandas and Numpy uses vectorization to speed up the task.

```python
# %%timeit can be used in ipython to calculate the time for opeartion.
%%timeit -n 100
summary = 0
for item in s:
  summary += item   # slow

%%timeit -n 100
summary = np.sum(s)   # faster
```

### Broadcasting

You can apply a value to every element of the series. Similar to map function.

```python
s += 2    # add 2 to all elements using broadcasting, much faster than the next method.
s.head()

for label, value in s.iteritems():
  s.set_value(label, value+2)   # add 2 to all elements.

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'   # we can mix types in series.

# We can also combine two series. We can also have multiple keys with same name.
sports = pd.Series({
  'Archery': 'Bhutan',
  'Golf': 'Scotland',
  'Sumo': 'Japan'
})
cricket_loving_countries = pd.Series(['Australia', 'India', 'South Africa'], index=['Cricket', 'Cricket', 'Cricket'])

all_countries = sports.append(cricket_loving_countries)   # all_countries contain both the series.
```

## DataFrame

It is largely used for cleaning the data. It is a table with multiple columns and index. Each column will have a label.
