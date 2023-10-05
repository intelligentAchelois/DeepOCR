#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import click
import numpy as np

from typing import Union, Dict
from tools.inference.utils import read_image
from tools.inference.text_recognition import TextRecognition
from tools.inference.text_detection import TextDetection


def draw_ocr(image_path: str, text_boxes: list, text_preds: list) -> np.ndarray:
    """Draw the predicted text boxes on the image."""
    image = read_image(image_path)
    h, _, _ = image.shape

    # Get height of the text in pixels to calculate the offset
    # using opencv font
    (text_height, _), _ = cv2.getTextSize("Text", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    num_text_boxes = len(text_boxes)

    for idx, (pts, text) in enumerate(zip(text_boxes, text_preds)):
        # Draw four points
        pts = pts.reshape(4, 2)
        cv2.polylines(image, [pts.astype(np.int32)], True, (0, 255, 0), 2)

        # Draw the predicted text
        x_min, y_min = np.min(pts, axis=0)

        cv2.putText(image, f"{idx}", (int(x_min), int(y_min) + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw the predicted text
        text_pos_y = h - (num_text_boxes - idx) * text_height
        cv2.putText(image, f"{idx}: {text}", (5, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image


class Pipeline:
    def __init__(self, det_model_path: str, rec_model_path: str, labels_file: str) -> None:
        self.rec_model_path = rec_model_path
        self.det_model_path = det_model_path
        self.labels_file = labels_file

        # Initialize the text detection model
        self.text_detection = TextDetection(self.det_model_path)

        # Initialize the text recognition model
        self.text_recognition = TextRecognition(
                                    self.rec_model_path,
                                    self.labels_file
                                )

    def get_bbox_crop(self, image, pts):
        """Crop the image to the bounding box."""
        # Do perspective transforms to get the bounding box
        pts = pts.reshape(4, 2)
        max_crop_width = max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3])
        )

        max_crop_height = max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2])
        )

        dst = np.array([
            [0, 0],
            [max_crop_width - 1, 0],
            [max_crop_width - 1, max_crop_height - 1],
            [0, max_crop_height - 1]], dtype='float32'
        )

        M = cv2.getPerspectiveTransform(pts, dst)

        crop = cv2.warpPerspective(
            image, M, (int(max_crop_width), int(max_crop_height)),
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )

        # rotate the crop based on height to width ratio
        crop_h, crop_w = crop.shape[:2]
        if crop_h*1.0 / crop_w > 1.5:
            crop = np.rot90(crop)

        return crop

    def predict(self, image: Union[str, np.ndarray]) -> Dict:
        """Predict the text in the image."""
        image = read_image(image)

        # Run the text detection model
        text_boxes, scores = self.text_detection.predict(image)

        # Run the text recognition model
        predicted_texts = []
        for i, box in enumerate(text_boxes):
            text_image = self.get_bbox_crop(image, box)

            text_pred, _, _ = self.text_recognition.predict(text_image)
            predicted_texts.append(text_pred)

        return text_boxes, scores, predicted_texts


@click.command()
@click.option('--det-model', default='models/text_det.onnx', help='Path to the text detection model.')
@click.option('--rec-model', default='models/text_rec.onnx', help='Path to the text recognition model.')
@click.option('--labels-file', default='configs/en.txt', help='Path to the labels file.')
@click.option('--data-path', default='samples', help='Path to the input image.')
@click.option('--output-path', default='output', help='Path to the output image.')
def main(det_model, rec_model, labels_file, data_path, output_path):

    # Check data_path is a directory or a file
    if os.path.isdir(data_path):
        image_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    else:
        image_files = [data_path]

    # Initialize the pipeline
    pipeline = Pipeline(det_model, rec_model, labels_file)

    # Predict the text in the image
    for image_path in image_files:
        print(f"Image path : {image_path}")
        text_boxes, scores, text_preds = pipeline.predict(image_path)

        # Print the predicted text
        print("Predictions:")
        for s, t in zip(scores, text_preds):
            s = float(s)
            print(f'  - {t} : {s:.5f}')

        print("\n")
        # Draw the predicted text boxes
        image = draw_ocr(image_path, text_boxes, text_preds)

        # Save the image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), rgb_image)


if __name__ == '__main__':
    main()
