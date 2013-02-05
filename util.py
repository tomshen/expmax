import time
import numpy as np

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def exportFile(filename, data):
    np.savetxt(filename, data)
    return filename

def importFile(filename):
    return np.genfromtxt(filename)