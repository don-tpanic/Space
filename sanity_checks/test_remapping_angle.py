import unittest
import numpy as np

def compute_single_heatmap_fields_info(heatmap, max_value_indices_in_clusters):
    angles = []
    center_y, center_x = np.array(heatmap.shape) // 2
    for max_index in max_value_indices_in_clusters:
        x, y = max_index
        dx = x - center_x
        dy = y - center_y 
        angle = np.degrees(np.arctan2(dy, dx))
        angles.append(angle)
    
    return angles


class TestComputeSingleHeatmapFieldsInfo(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.heatmap = np.random.rand(17, 17)
        self.max_value_indices_in_clusters = [[0, 8], [8, 16], [16, 0], [16, 16], [9, 7], [8, 9], [0, 0]]
    
    def test_angles_and_mean_angle(self):
        expected_angles = [180.0, 90.0, -45.0, 45.0, -45.0, 90, -135]
        
        angles = compute_single_heatmap_fields_info(self.heatmap, self.max_value_indices_in_clusters)
        for angle, expected_angle in zip(angles, expected_angles):
            self.assertAlmostEqual(angle, expected_angle, places=1)


if __name__ == '__main__':
    unittest.main()
