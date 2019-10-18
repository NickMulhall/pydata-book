from numpy.random import randn
import numpy as np
np.random.seed(123)
import os
import matplotlib.pyplot as plt
import pandas as pd
import json

path = 'datasets/bitly_usagov/example.txt'
records = [json.loads(line) for line in open(path)]
print(records[0])

# fails as not all records have a 'tz' field
#time_zones = [rec['tz'] for rec in records]

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print(time_zones[10]) #print 10 time zones

# Produce counts by time zone. 2 methods. 1 hard way using the std python lib and 2. easier using pandas

#1
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

counts = get_counts(time_zones)
print(counts['America/New_York'])

#2
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int) #values will initialise to 0
    for x in sequence:
        counts[x] += 1
    return counts

counts = get_counts2(time_zones)
print(counts['America/New_York'])

# Get the 10 most common time zones
from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))

#doing this with Pandas
import pandas as pd
frame = pd.DataFrame(records)
#print(frame.info)

tz_counts = frame['tz'].value_counts()
#print(tz_counts[:10])

#fill the missing or blank time zones and viualise using matplotlib
#Munging / wrangling = convert from raw to another format that allows for more convenient consumption of data

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])
import seaborn as sns
subset = tz_counts[:10]
sns.barplot(y=subset.index, x=subset.values)

#plot example
from math import radians
import numpy as np
def showit():
    x = np.arange(0, radians(1000), radians(12))
    plt.plot(x, np.cos(x), 'b')
    plt.show()
showit()






