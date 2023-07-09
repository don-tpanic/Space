import numpy as np
import matplotlib.pyplot as plt



# r is a gaussian with mean 0.5 std 0.1
r = np.random.normal(0.5, 0.1, 100)
theta = np.linspace(0, 2*np.pi, len(r))

# plot a polar plot such that
# the top is 0 degrees, 
# the right is 90 degrees
# the bottom is 180 degrees
# the left is 270 degrees

fig = plt.figure()
# plot two subplots side by side first is regular second is polar
ax = fig.add_subplot(121)
ax.plot(theta, r, 'bo')

ax = fig.add_subplot(122, polar=True)
ax.plot(theta, r, 'bo')

# keep grid but remove radius labels
ax.set_rgrids([0.25, 0.5, 0.75, 1], labels=['', '', '', ''])

# remove theta labels
ax.set_thetagrids([0, 90, 180, 270], labels=['', '', '', ''])


plt.savefig('test.png')