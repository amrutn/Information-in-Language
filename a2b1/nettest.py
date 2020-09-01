import numpy as np


def generateData(datasize, marginoferror):
    first = int(datasize/2)
    second = int(datasize/2)
    sizeone = (first,1)
    sizetwo = (second,1)
    #marginoferror = 0.01

    agen = np.random.lognormal(0, 1, sizeone)
    bgen = np.random.lognormal(0, 1, sizetwo)
    cgen = np.zeros_like(agen)
    boolArrgen = (agen**2+bgen**2-1<=marginoferror)
    cgen[boolArrgen] = 1

    atrues = np.random.random(sizetwo)
    btruesquares = 1-atrues**2 + (2*np.random.random()-1) * marginoferror
    underZeroBool = btruesquares < 0
    btruesquares[underZeroBool] = 0
    btrues = np.sqrt(btruesquares)
    boolArrtrues = (atrues**2+btrues**2-1 <= marginoferror)
    ctrues = np.zeros_like(atrues)
    ctrues[boolArrtrues] = 1

    afin = np.concatenate((agen, atrues), axis = 0)
    bfin = np.concatenate((bgen, btrues), axis = 0)
    cfin = np.concatenate((cgen, ctrues), axis = 0)

    x = np.array([afin, bfin])
    y = cfin
    print(x.shape, y.shape)
    return (x,y)
