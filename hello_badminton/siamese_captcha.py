# -*- coding: utf-8 -*-

# File: siamese_captcha.py
# License: MIT License
# Copyright: (c) 2023 Jungheil <jungheilai@gmail.com>
# Created: 2023-11-03
# Brief:
# --------------------------------------------------

import base64
from io import BytesIO
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


class SiameseCaptcha:
    def __init__(self):
        self._detector_session = ort.InferenceSession("resources/detector.onnx")
        self._siamese_session = ort.InferenceSession("resources/siamese.onnx")

    def _get_chaptcha_chars(self, img_base64: str) -> Tuple[list, list]:
        image_bytes = base64.b64decode(img_base64)
        img = Image.open(BytesIO(image_bytes))
        poses = self._detector_predict(img)
        chars = []
        centre = []

        for box in poses:
            x1, y1, x2, y2 = box
            char = img.crop((x1, y1, x2, y2))
            char = char.resize((28, 28))
            chars.append(char)
            centre.append({"x": (x1 + x2) // 2, "y": (y1 + y2) // 2})
        return chars, centre

    def _get_chinese_char(self, text):
        font_path = "resources/fzhei.ttf"

        image = Image.new("RGB", (28, 28), color="white")
        font = ImageFont.truetype(font_path, 28)
        draw = ImageDraw.Draw(image)

        draw.text((0, -2), text, font=font, fill="black")
        return image

    def _detector_preproc(self, img, input_size):
        r = min(input_size[0] / img.size[0], input_size[1] / img.size[1])

        img = img.resize((int(img.size[0] * r), int(img.size[1] * r)))
        img = np.array(img).astype(np.uint8)
        img = img[..., ::-1]
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        padded_img[: img.shape[0], : img.shape[1]] = img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _detector_postproc(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        def _nms(boxes, scores, nms_thr):
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= nms_thr)[0]
                order = order[inds + 1]

            return keep

        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = _nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [
                    valid_boxes[keep],
                    valid_scores[keep, None],
                    valid_cls_inds[keep, None],
                ],
                1,
            )
        return dets

    def _detector_predict(self, img):
        im, ratio = self._detector_preproc(img, (416, 416))
        ort_inputs = {self._detector_session.get_inputs()[0].name: im[None, :, :, :]}
        output = self._detector_session.run(None, ort_inputs)
        predictions = self._detector_postproc(output[0], (416, 416))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        pred = self._multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        try:
            final_boxes = pred[:, :4].tolist()
            result = []
            for b in final_boxes:
                if b[0] < 0:
                    x_min = 0
                else:
                    x_min = int(b[0])
                if b[1] < 0:
                    y_min = 0
                else:
                    y_min = int(b[1])
                if b[2] > img.size[0]:
                    x_max = int(img.size[0])
                else:
                    x_max = int(b[2])
                if b[3] > img.size[1]:
                    y_max = int(img.size[1])
                else:
                    y_max = int(b[3])
                result.append([x_min, y_min, x_max, y_max])
        except Exception:
            return []
        return result

    def _siamese_preproc(self, img):
        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
        return img

    def _siamese_predict(self, input_0, input_1):
        output = self._siamese_session.run(
            None, {"input_0": input_0, "input_1": input_1}
        )
        return output[0]

    def __call__(
        self,
        img_base64: str,
        words: list,
        **kwargs,
    ) -> list:
        words_len = len(words)
        chaptcha_chars, points = self._get_chaptcha_chars(img_base64)
        if len(chaptcha_chars) < words_len:
            return []
        target_chars = [self._get_chinese_char(i) for i in words]

        chaptcha_chars = [self._siamese_preproc(i) for i in chaptcha_chars]
        target_chars = [self._siamese_preproc(i) for i in target_chars]

        input_0 = np.zeros(
            (words_len * len(chaptcha_chars), 3, 28, 28), dtype=np.float32
        )
        input_1 = np.zeros(
            (words_len * len(chaptcha_chars), 3, 28, 28), dtype=np.float32
        )
        idx = 0
        for t in target_chars:
            for c in chaptcha_chars:
                input_0[idx] = t
                input_1[idx] = c
                idx += 1
        output = self._siamese_predict(input_0, input_1)
        output = output.reshape((words_len, -1))
        _, c = linear_sum_assignment(output)
        return [points[i] for i in c]
