import unittest

from backend.object_detection import ObjectTemporalConfirmer, associate_person, normalize_dangerous_label


class ObjectTemporalConfirmerTests(unittest.TestCase):
    def setUp(self):
        self.confirmer = ObjectTemporalConfirmer(required_frames=3)
        self.item = {'label': 'knife', 'confidence': 82.0, 'box': [10, 10, 12, 20]}

    def test_requires_three_consecutive_matching_frames(self):
        self.assertFalse(self.confirmer.update('u1', [self.item])[0]['confirmed'])
        self.assertFalse(self.confirmer.update('u1', [self.item])[0]['confirmed'])
        third = self.confirmer.update('u1', [self.item])[0]
        self.assertTrue(third['confirmed'])
        self.assertEqual(third['confirmationFrames'], 3)
        self.assertEqual(third['objectTrackId'], self.confirmer.update('u1', [self.item])[0]['objectTrackId'])

    def test_empty_frame_resets_confirmation(self):
        self.confirmer.update('u1', [self.item])
        self.confirmer.update('u1', [self.item])
        self.assertEqual([], self.confirmer.update('u1', []))
        restarted = self.confirmer.update('u1', [self.item])[0]
        self.assertEqual(1, restarted['confirmationFrames'])
        self.assertFalse(restarted['confirmed'])

    def test_class_change_does_not_confirm(self):
        self.confirmer.update('u1', [self.item])
        changed = self.confirmer.update('u1', [{**self.item, 'label': 'gun'}])[0]
        self.assertEqual(1, changed['confirmationFrames'])

    def test_ignores_unknown_and_safe_classes(self):
        self.assertEqual([], self.confirmer.update('u1', [{'label': 'metallic object', 'box': [0, 0, 10, 10]}]))
        self.assertEqual([], self.confirmer.update('u1', []))

    def test_multiple_tracks_and_person_association(self):
        detections = self.confirmer.update('u1', [self.item, {**self.item, 'box': [75, 70, 10, 12]}])
        self.assertNotEqual(detections[0]['objectTrackId'], detections[1]['objectTrackId'])
        people = [{'box': [0, 0, 35, 45]}, {'box': [65, 60, 30, 35]}]
        self.assertEqual('person_02', associate_person(detections[1], people))
        self.assertIsNone(associate_person(detections[0], []))

    def test_normalizes_knife_labels(self):
        self.assertEqual('knife', normalize_dangerous_label(' Knife '))
        self.assertEqual('knife', normalize_dangerous_label('kitchen_knife'))


if __name__ == '__main__':
    unittest.main()
