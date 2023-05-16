import numpy as np
import matplotlib.pyplot as plt

"""
In 2D unity env28, the agent moves from (-4,-4) to (-4,4) 
along the z-axis and the location reps are collected along 
the way. 

So to plot heatmap of every single unit onto the 2D space,
we want to make sure that we plot the unit's output
at the correct location.

One way is to record the x-z coordinates and plot them 
as scatter plot and the cmap will be the output of the 
unit across these coordinates.

Another way is to reshape the output of the unit into
a 2D array and plot it as a heatmap.

This test is to see if two approaches are equivalent.
"""

# we simpify the env to be 2x2 (each row is a location)
# and we simplify model unit to be 3.
# (n_loc, n_unit)
model_reps = np.array(
   [
        [0.1, 0.2, 0.3],
        [0.3, 0.7, 0.8],
        [0.7, 0.4, 0.5],
        [1.0, 0.9, 0.10],
   ]
)

# we only use the first unit
model_rep_1 = model_reps[:, 0]
# so the ground truth is that the agent's location-unit output
# in order should be 0.1, 0.3, 0.7, 1.0
# which if translate to 2D coordinates, can be the below.
x_coords = np.array([-4, -4, -3, -3])
z_coords = np.array([-4, -3, -4, -3])

# --------------------------------------------------------------------------------
# CHECK1: visual -  plot using two approaches each as a subplot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(x_coords, z_coords, c=model_rep_1, cmap='viridis', label='scatter', s=500)
# plot the cmap values on top of the scatter plot
for i in range(len(x_coords)):
    axs[0].text(x_coords[i], z_coords[i], round(model_rep_1[i], 1))

axs[1].imshow(model_rep_1.reshape((2, 2)), cmap='viridis', label='reshape')
# plot the cmap values on top of the heatmap
for i in range(len(x_coords)):
    axs[1].text(i%2, i//2, round(model_rep_1[i], 1))
plt.savefig('tests/testReshape_forHeatMap.png')
# conclusion: we show that two approaches results are not the same but rotated.
# we can further confirm this by comparing the values directly (CHECK2)

# --------------------------------------------------------------------------------
# CHECK2: numerical - compare values
# if we plot as scatter using coordinates, we expect to 
# transform [0.1, 0.3, 0.7, 1.0] to 
# [[0.3, 1.0],
# [0.1, 0.7]]

# if we plot from reshape, we get different reult
# print(model_rep_1.reshape((2, 2)))
# [[0.1 0.3]
#  [0.7 1. ]]

# so if we still want to use reshape (as then we can use to imshow to plot heatmap nicely),
# we need to rotate the array by 90 degrees.
print(np.rot90(model_rep_1.reshape((2, 2))))
