import numpy as np
import matplotlib.pyplot as plt
from util import importFile, exportFile

def scatterPlot(data):
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Data plotted on map of the world')
    im = plt.imread('world-large.png')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.imshow(im, extent=[-180, 180, -90, 90])
    plt.plot(data[0], data[1], 'ro')
    plt.show()

def main():
    data = importFile('toronto_data.txt')
    scatterPlot([data[1], data[0]])
    
if __name__ == "__main__":
    main()