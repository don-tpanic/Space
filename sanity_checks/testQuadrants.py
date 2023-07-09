import matplotlib.pyplot as plt


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


# second quadrant:
# sample coords by return index of coords with -4 =< x < 0 and 0 <= y < 4
sample_indices = [i for i, x in enumerate(x_axis_coords) if -4 <= x < 0 and 0 <= y_axis_coords[i] < 4]
print(sample_indices)

# plot the sample coords
fig, ax = plt.subplots()
# set lims to match the coords
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.scatter([x_axis_coords[i] for i in sample_indices], [y_axis_coords[i] for i in sample_indices])
plt.savefig('sample_coords.png')