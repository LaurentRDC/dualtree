
from dualtree import dualtree, idualtree
import numpy as n

if __name__ == '__main__':
    signal = n.random.rand(1e4, 1e4)
    transf = dualtree(signal, level = 'max')
    recon = idualtree(transf)