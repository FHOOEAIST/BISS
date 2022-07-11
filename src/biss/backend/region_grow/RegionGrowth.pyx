# -*- coding: utf-8 -*-
from collections import deque
cimport numpy as np
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

cdef class RegionGrow3D:
    cdef np.uint16_t[:,:,:] images
    cdef np.uint8_t[:,:,:] outputMask

    cdef int sx, sy, sz
    cdef int upperThreshold
    cdef int lowerThreshold
    cdef neighborMode
    cdef queue
    
    def __cinit__(self, np.uint16_t[:,:,:] images, int upperThreshold, int lowerThreshold):
        self.images = images
        
        self.sz = images.shape[0]
        self.sy = images.shape[1]
        self.sx = images.shape[2]

        self.upperThreshold = upperThreshold
        self.lowerThreshold = lowerThreshold
        self.queue = deque()
    
    def apply(self, seeds, outMask, update=False):
        """
        seed: list of (z,y,x)
        """
        self.outputMask = outMask

        cdef int newItem[3]
        for seed in seeds:
            newItem = seed
            self.outputMask[newItem[0], newItem[1], newItem[2]] = 1
            self.queue.append((seed[0], seed[1], seed[2]))

        while len(self.queue) != 0:
            newItem = self.queue.pop()
            neighbors = self.getNeighbors(newItem)
            for neighbor in neighbors:
                self.checkNeighbour(neighbor[0], neighbor[1], neighbor[2])
        return self.outputMask


    cdef int[:,:] getNeighbors(self, int[:] newItem):
        neighbors = [
                [newItem[0]-1, newItem[1], newItem[2]],
                [newItem[0]+1, newItem[1], newItem[2]],
                [newItem[0], newItem[1]-1, newItem[2]],
                [newItem[0], newItem[1]+1, newItem[2]],
                [newItem[0], newItem[1], newItem[2]-1],
                [newItem[0], newItem[1], newItem[2]+1],
            ]
        return np.array(neighbors, dtype=np.int32)

    cdef checkNeighbour(self, int z, int y, int x):
        cdef int intensity
        if (x < self.sx and y < self.sy and z < self.sz 
            and x > -1 and y > -1 and z > -1):
            intensity = self.images[z, y, x]
            if self.isIntensityAcceptable(intensity) and self.outputMask[z,y,x] == 0:
                self.outputMask[z,y,x] = 1
                self.queue.append((z, y, x))
    
    cdef isIntensityAcceptable(self, int intensity):
        if self.lowerThreshold <= intensity <= self.upperThreshold:
            return True
        return False
