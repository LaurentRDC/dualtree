from dualtree import baseline
import numpy as n

array = n.sin(n.linspace(0, 10, num = 1000))
test = baseline(array, max_iter = 100)