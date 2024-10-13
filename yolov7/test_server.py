import unittest
import requests
import json
import os


class TestNavForBlindServer(unittest.TestCase):
    # Update if your server is running elsewhere
    SERVER_URL = 'http://localhost:5000'

    def test_navigation_valid_image(self):
        """Test the /navigation endpoint with a valid image."""
        url = f'{self.SERVER_URL}/navigation'
        image_path = 'navigation_test.jpg'  # Replace with your valid test image path

        # Open the image file in binary mode
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(url, files=files)

        self.assertEqual(response.status_code, 200, "Expected status code 200")

        data = response.json()
        self.assertIn('path_suggestion', data,
                      "Response should contain 'path_suggestion'")
        self.assertIn('scene_label', data,
                      "Response should contain 'scene_label'")
        self.assertIn('detected_objects', data,
                      "Response should contain 'detected_objects'")

        # Check types of the returned values
        self.assertIsInstance(data['path_suggestion'], (str, type(
            None)), "path_suggestion should be a string or None")
        self.assertIsInstance(data['scene_label'],
                              str, "scene_label should be a string")
        self.assertIsInstance(data['detected_objects'],
                              list, "detected_objects should be a list")

    def test_text_recognition_valid_image(self):
        """Test the /text_recognition endpoint with a valid image containing text."""
        url = f'{self.SERVER_URL}/text_recognition'
        # Replace with your valid text image path
        image_path = 'text_recognition_test.jpg'

        # Open the image file in binary mode
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(url, files=files)

        self.assertEqual(response.status_code, 200, "Expected status code 200")

        data = response.json()
        self.assertIn('recognized_text', data,
                      "Response should contain 'recognized_text'")
        self.assertIsInstance(data['recognized_text'],
                              str, "recognized_text should be a string")
        self.assertTrue(len(data['recognized_text']) >
                        0, "recognized_text should not be empty")

    def test_navigation_invalid_image(self):
        """Test the /navigation endpoint with an invalid image."""
        url = f'{self.SERVER_URL}/navigation'
        image_path = 'invalid_image.jpg'  # Replace with your invalid image path

        # Create an invalid image file (e.g., empty or corrupted)
        with open(image_path, 'wb') as image_file:
            image_file.write(b'this is not a valid image file')

        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(url, files=files)

        # Clean up the invalid image file
        os.remove(image_path)

        # Expecting a 500 Internal Server Error or a custom error status code
        self.assertNotEqual(response.status_code, 200,
                            "Expected an error status code")
        data = response.json()
        self.assertIn('error', data, "Response should contain 'error'")

    def test_text_recognition_no_image(self):
        """Test the /text_recognition endpoint without sending an image."""
        url = f'{self.SERVER_URL}/text_recognition'
        files = {}  # No image provided

        response = requests.post(url, files=files)

        self.assertEqual(response.status_code, 400,
                         "Expected status code 400 for bad request")
        data = response.json()
        self.assertIn('error', data, "Response should contain 'error'")
        self.assertEqual(data['error'], 'No image provided',
                         "Error message should indicate missing image")


if __name__ == '__main__':
    unittest.main()
