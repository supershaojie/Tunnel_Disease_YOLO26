from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from shared.b19.augment import BaseTransform


class LetterBox(BaseTransform):
    """Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates corresponding
    labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scale_fill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            padding_value (int): Value for padding the image. Default is 114.
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels: dict[str, Any] | None = None, image: np.ndarray = None) -> dict[str, Any] | np.ndarray:
        """Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (dict[str, Any] | None): A dictionary containing image data and associated labels, or empty dict if
                None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (dict[str, Any] | np.ndarray): If 'labels' is provided, returns an updated dictionary with the resized and
                padded image, updated labels, and additional metadata. If 'labels' is empty, returns the resized and
                padded image only.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        return_image_only = len(labels) == 0
        if image is not None:
            labels["img"] = image
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not return_image_only:
            labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        if return_image_only:
            return labels["img"]
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute letterboxing parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary containing 'img'.

        Returns:
            (dict): Parameters including 'orig_shape', 'new_shape', 'ratio', padding, and resize info.
        """
        img = labels["img"]
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)

        return {
            "orig_shape": shape,
            "new_shape": new_shape,
            "ratio": ratio,
            "new_unpad": new_unpad,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Resize and pad the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict): Parameters from get_params.

        Returns:
            (dict): Updated labels with resized and padded image.
        """
        img = labels["img"]
        shape = img.shape[:2]
        new_unpad = params["new_unpad"]

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        h, w, c = img.shape
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        labels["img"] = img
        labels["resized_shape"] = params["new_shape"]
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply letterboxing to semantic segmentation mask.

        Args:
            labels (dict[str, Any]): Dictionary containing 'semantic_mask'.
            params (dict): Parameters from get_params.

        Returns:
            (dict): Updated labels with resized and padded semantic mask.
        """
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        mask = labels["semantic_mask"]
        shape = params["orig_shape"]
        new_unpad = params["new_unpad"]
        if shape[::-1] != new_unpad:
            mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_NEAREST)
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        labels["semantic_mask"] = mask
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Update instance coordinates after letterboxing.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances'.
            params (dict): Parameters from get_params.

        Returns:
            (dict): Updated labels with transformed instances.
        """
        if "instances" in labels:
            labels = self._update_labels(labels, params["ratio"], params["left"], params["top"], params["orig_shape"])
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (params["left"], params["top"]))  # for evaluation
        return labels

    @staticmethod
    def _update_labels(
        labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float, orig_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """Update labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels to account for resizing and padding
        applied during letterboxing.

        Args:
            labels (dict[str, Any]): A dictionary containing image labels and instances.
            ratio (tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.
            orig_shape (tuple[int, int]): Original image shape (height, width) before resizing.

        Returns:
            (dict[str, Any]): Updated labels dictionary with modified instance coordinates.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh, (480, 640))
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*orig_shape[::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels
