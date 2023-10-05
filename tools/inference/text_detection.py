# -*- coding:utf-8 -*-

import numpy as np
import onnxruntime as ort

from typing import Union, Dict
from .utils import read_image, pad_image, \
    resize_with_side_limit, quad_points_from_bitmap

MAX_SIDE_LEN = 960
PAD_VALUE = 0
MIN_INPUT_SIZE = 32

# Bounding post processing parameters
MAX_BBOXES = 1000
MIN_BBOX_SIZE = 3
BOX_THRES = 0.6
MASK_THRES = 0.3
CLIPPING_RATIO = 1.5


class TextDetection:
    def __init__(self, model_path) -> None:

        self.model_path = model_path

        # Load the model ONNX model
        self.model = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def predict(self, image: Union[str, np.ndarray]) -> Dict:
        """ Predict the text in the image."""
        image = read_image(image)
        input_h, input_w, _ = image.shape
        input, ratio_h, ratio_w = self.preprocess(image)

        # Run the prediction
        # Prepare batch
        input = np.expand_dims(input, axis=0).astype(np.float32)
        output = self.model.run([self.output_name], {self.input_name: input})

        # Postprocess the output
        boxes, scores = self.postprocess(output[0], (input_h, input_w))
        return boxes, scores

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image."""
        image = image.astype(np.float32)
        image = pad_image(image, MIN_INPUT_SIZE, PAD_VALUE)
        image, ratio_h, ratio_w = resize_with_side_limit(image, MAX_SIDE_LEN, 32)

        # mean, std normalization (these vectors are in shape (1,1,3))
        mean_vector = np.array([[[0.485, 0.456, 0.406]]])
        std_vector = np.array([[[0.229, 0.224, 0.225]]])

        image = (image / 255.0 - mean_vector) / std_vector
        image = image.transpose((2, 0, 1))
        return image, ratio_h, ratio_w

    def order_points_in_clockwise(self, pts):
        """Order the points in clockwise."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tpoints = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tpoints), axis=1)
        rect[1] = tpoints[np.argmin(diff)]
        rect[3] = tpoints[np.argmax(diff)]
        return rect

    def postprocess(self, preds: np.ndarray, input_shape: tuple) -> Dict:
        """Postprocess the output."""
        pred = preds[0][0]
        input_h, input_w = input_shape
        mask = pred > MASK_THRES
        points_list, scores = quad_points_from_bitmap(
                            pred, mask, input_w, input_h,
                            MAX_BBOXES, MIN_BBOX_SIZE,
                            BOX_THRES, CLIPPING_RATIO
                        )

        # order bbox points in clockwise
        boxes = []
        for points in points_list:
            rect = self.order_points_in_clockwise(points)

            # Clip the bounding box to the image
            rect[:, 0] = np.clip(rect[:, 0], 0, input_w - 1)
            rect[:, 1] = np.clip(rect[:, 1], 0, input_h - 1)

            box_width = np.linalg.norm(rect[0] - rect[1])
            box_height = np.linalg.norm(rect[0] - rect[3])

            if min(box_width, box_height) < MIN_BBOX_SIZE:
                continue

            boxes.append(rect)

        return boxes, scores
