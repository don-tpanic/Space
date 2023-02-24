import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def test_plotting_order():
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    multiplier = 1
    x_axis_coords = []
    y_axis_coords = []
    for i in range(x_min*multiplier, x_max*multiplier+1):
        for j in range(y_min*multiplier, y_max*multiplier+1):
            x_axis_coords.append(i/multiplier)
            y_axis_coords.append(j/multiplier)

    components = range(len(x_axis_coords))[::-1]
    print(list(components))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_axis_coords, y_axis_coords, c=components, cmap='viridis')
    plt.savefig('testScatter.png')


def test_plotting_normalize():
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4
    multiplier = 2
    x_axis_coords = []
    y_axis_coords = []
    for i in range(x_min*multiplier, x_max*multiplier+1):
        for j in range(y_min*multiplier, y_max*multiplier+1):
            x_axis_coords.append(i/multiplier)
            y_axis_coords.append(j/multiplier)
    fig, ax = plt.subplots(1, 2)

    components = range(len(x_axis_coords))[::-1]
    print(f'num components: {len(components)}')  # 289

    cmap = cm.get_cmap('viridis')

    # print(cmap(0))
    # print(cmap(1))
    # print(cmap(2))
    # print(cmap(3))
    # print(cmap(4))
    # print(cmap(5))
    # print(cmap(999))
    
    ax[0].scatter(x_axis_coords, y_axis_coords, c=components, cmap='viridis')

    print(f'max: {np.max(components)}')
    component_max = 333
    component_min = np.min(components)
    components = (components - component_min) / (component_max - component_min)
    c = []
    for i in range(len(components)):
        c.append(cmap(components[i]))
    ax[1].scatter(x_axis_coords, y_axis_coords, c=c)
    plt.savefig('testScatter.png')


if __name__ == '__main__':
    # test_plotting_order()
    test_plotting_normalize()