# -*- coding: utf-8 -*-

# File: captcha.py
# License: MIT License
# Copyright: (c) 2023 Jungheil <jungheilai@gmail.com>
# Created: 2023-11-15
# Brief:
# --------------------------------------------------

import base64
import inspect
import threading
from types import CellType

import cv2
import numpy as np

from hello_badminton.siamese_captcha import SiameseCaptcha
from hello_badminton.utils.timeout_decorator import enforce_timeout

__all__ = ["captcha_registry"]


class CaptchaRegistry:
    def __init__(self):
        self._captcha = {}

    @property
    def captcha(self):
        return self._captcha

    def register(self, name: str):
        def wrapper(func: CellType):
            assert (
                "img_base64" in inspect.signature(func).parameters.keys()
                and "words" in inspect.signature(func).parameters.keys()
            ), "The function must have 'img_base64' and 'words' parameters."
            self._captcha[name] = func
            return func

        return wrapper

    def get(self, name):
        return self._captcha.get(name)

    def remove(self, name):
        return self._captcha.pop(name, None)


captcha_registry = CaptchaRegistry()

manual_lock = threading.Lock()
manual_idx = 0


@captcha_registry.register("manual")
def captcha_points_manual(img_base64: str, words: list, **kwargs) -> list:
    with manual_lock:
        global manual_idx
        idx = manual_idx
        manual_idx += 1
    timeout = kwargs.get("timeout", None)

    @enforce_timeout(timeout, False)
    def _show_captcha(img, idx=None):
        point = []

        def _on_mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                point.append({"x": x, "y": y})
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        window_name = f"captcha-{idx}" if idx is not None else "captcha"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _on_mouse_cb)
        while True:
            cv2.imshow(window_name, img)
            key = cv2.waitKey(10)
            if len(point) == len(words) or key == ord("q"):
                cv2.destroyWindow(window_name)
                break
        return point

    img_data = base64.b64decode(img_base64)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)

    print(f"{idx}: {words}")
    ret = _show_captcha(img, idx)

    return ret


siamese_captcha = SiameseCaptcha()
captcha_registry.register("siamese")(siamese_captcha)
