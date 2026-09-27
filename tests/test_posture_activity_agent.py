import unittest

from backend.posture_activity_agent import PostureActivityAgent


def standing_points():
    coords = {
        5: (45, 15), 6: (55, 15),
        11: (45, 35), 12: (55, 35),
        13: (45, 50), 14: (55, 50),
        15: (45, 65), 16: (55, 65),
    }
    return [{'index': index, 'x': x, 'y': y, 'confidence': 90} for index, (x, y) in coords.items()]


def sitting_points():
    coords = {
        5: (45, 15), 6: (55, 15),
        11: (45, 35), 12: (55, 35),
        13: (45, 42), 14: (55, 42),
        15: (55, 42), 16: (45, 42),
    }
    return [{'index': index, 'x': x, 'y': y, 'confidence': 90} for index, (x, y) in coords.items()]


class PostureActivityAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = PostureActivityAgent()

    def test_extended_legs_resolve_to_standing(self):
        result = self.agent.classify([10, 10, 20, 70], standing_points())
        self.assertEqual(result['activity'], 'STANDING')
        self.assertEqual(result['source'], 'pose_joint_geometry')

    def test_bent_legs_and_low_hip_resolve_to_sitting(self):
        result = self.agent.classify([10, 10, 40, 45], sitting_points())
        self.assertEqual(result['activity'], 'SITTING')
        self.assertEqual(result['source'], 'pose_joint_geometry')

    def test_box_shape_fallback_still_returns_one_of_the_two_labels(self):
        sitting = self.agent.classify([10, 10, 40, 45], [])
        standing = self.agent.classify([10, 10, 20, 70], [])
        self.assertEqual(sitting['activity'], 'SITTING')
        self.assertEqual(standing['activity'], 'STANDING')


if __name__ == '__main__':
    unittest.main()
