import numpy as np 

"""
Confirmed, new PD proportion in many items changes add up.
"""

orig_PBD = 7/100
orig_PB = 8.7/100
orig_inactive = 5/100
orig_PD = 55.1/100
orig_P = 5.2/100
orig_B = 1.6/100
orig_D = 15.8/100
orig_noType = 1.6/100
sum_ = sum([orig_PBD, orig_PB, orig_inactive, orig_PD, orig_P, orig_B, orig_D, orig_noType])
assert np.isclose(sum_, 1.0), f"Sum: {sum_}"

# many items changes
PBD_to_PD = 49.1/100
PB_to_PD = 19.9/100
PD_to_PD = 82.7/100
P_to_PD = 31.8/100
B_to_PD = 3.1/100
D_to_PD = 47.8/100
noType_to_PD = 1.5/100

pct_PD = (PBD_to_PD * orig_PBD + PB_to_PD * orig_PB + PD_to_PD * orig_PD + P_to_PD * orig_P + B_to_PD * orig_B + D_to_PD * orig_D + noType_to_PD * orig_noType)
print(pct_PD)