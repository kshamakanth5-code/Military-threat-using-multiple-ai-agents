import unittest

from PIL import Image

from backend.facial_expression import analyze_facial_expression, predict_next_actions


class FacialExpressionTests(unittest.TestCase):
    def test_blank_frame_reports_unknown_emotion(self):
        report = analyze_facial_expression(Image.new('RGB', (64, 64)))
        self.assertEqual(report['emotion'], 'unknown')
        self.assertEqual(report['emotionConfidence'], 0)
        self.assertFalse(report['faceDetected'])

    def test_next_action_prediction_is_ranked(self):
        report = {'emotion': 'neutral', 'confidence': 50}
        prediction = predict_next_actions(report, 'walking', 'MEDIUM', 0)
        self.assertEqual(prediction['topPrediction'], prediction['rankedActions'][0]['action'])
        self.assertEqual(len(prediction['rankedActions']), 5)
        self.assertIn('decisionPolicy', prediction)


if __name__ == '__main__':
    unittest.main()