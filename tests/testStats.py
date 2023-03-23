import numpy as np
from scipy import stats 


samples = np.random.normal(0, 10, 1000)

# sem 
# print(stats.sem(samples))

# se of bootstrap
res = stats.bootstrap((samples,), np.mean)
print(np.mean(samples))
print(res.standard_error)
ci_low, ci_high = res.confidence_interval
print(ci_low, ci_high)