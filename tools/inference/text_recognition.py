# -*- coding:utf-8 -*-

import numpy as np
import onnxruntime as ort

from typing import Union, Dict
from .utils import read_image, resize_with_aspect_ratio


class TextRecognition:
    def __init__(self, model_path: str, labels_file: str) -> None:
        self.model_path = model_path
        self.labels_file = labels_file

        with open(self.labels_file, 'r', encoding='utf-8') as f:
            self.characters = [val.strip("\n").strip("\r\n") for val in f.read().splitlines()]

        # Load the model ONNX model
        self.model = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = [3, 48, 320]

    def predict(self, image: Union[str, np.ndarray]) -> Dict:
        """Predict the text in the image."""

        image = read_image(image)
        input = self.preprocess(image)
        input = np.expand_dims(input, axis=0).astype(np.float32)

        # Run the prediction
        # Output : [(batch_size, 40, 97)] Since we are sending single image batch_size = 1
        output = self.model.run([self.output_name], {self.input_name: input})

        return self.postprocess(output[0])

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image."""
        # Resize to the input shape and preserve aspect ratio
        input_image = np.zeros((self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        image_resized, _, _ = resize_with_aspect_ratio(image, self.input_shape[2], self.input_shape[1])

        # Normalize the image
        image_resized = image_resized.astype(np.float32)
        image_resized = image_resized.transpose((2, 0, 1))
        image_resized /= 255.0
        image_resized -= 0.5
        image_resized /= 0.5

        # Pad the image
        input_image[:, :image_resized.shape[1], :image_resized.shape[2]] = image_resized
        return input_image

    def postprocess(self, output: np.ndarray) -> str:
        """Postprocess the image."""

        # Get the prediction
        prediction = output[0]
        pred_argmax = np.argmax(prediction, axis=1)
        pred_probs = np.max(prediction, axis=1)

        # Convert the prediction to text
        text = ''
        char_probabilities = []
        softmax_values = []

        for i in range(len(pred_argmax)):
            if pred_argmax[i] != 0 and (not (i > 0 and pred_argmax[i - 1] == pred_argmax[i])):
                text += self.characters[pred_argmax[i]]
                char_probabilities.append(pred_probs[i])
                softmax_values.append(prediction[i])

        return text, char_probabilities, softmax_values
