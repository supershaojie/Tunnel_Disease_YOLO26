# Frozen from B19 4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6, AGPL-3.0.
from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any
import cv2
import numpy as np
from .instance import Instances
from .metrics import bbox_ioa
from .ops import segment2box


class BaseTransform:
    """Base class for image transformations in the Ultralytics library.

    This class provides a unified interface for applying transformations to images, object instances, and semantic
    segmentation masks. Subclasses should override `apply_image`, `apply_instances`, and/or `apply_semantic` for simple
    transforms, or override `__call__` directly for complex transforms that need shared state between image and
    annotation modifications.

    Methods:
        get_params: Compute transformation parameters shared across image, instances, and semantic mask.
        apply_image: Apply transformation to the image in labels['img'].
        apply_instances: Apply transformation to object instances in labels['instances'].
        apply_semantic: Apply transformation to semantic mask in labels['semantic_mask'].
        __call__: Orchestrate the transformation pipeline.
    """

    def __call__(self, labels):
        """Apply transformation to labels dict.

        Args:
            labels (dict): Dictionary containing 'img', optionally 'instances' and 'semantic_mask'.

        Returns:
            (dict): Transformed labels dictionary.
        """
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        return labels

    def get_params(self, labels):
        """Compute and return transformation parameters.

        This method allows sharing random state or computed matrices (e.g. affine matrix, flip
        decision) between image, instances, and semantic mask transformations.

        Args:
            labels (dict): Input labels dictionary.

        Returns:
            (dict): Parameters to pass to apply_image, apply_instances, and apply_semantic.
        """
        return {}

    def apply_image(self, labels, params=None):
        """Apply transformation to image.

        Args:
            labels (dict): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels

    def apply_instances(self, labels, params=None):
        """Apply transformation to object instances.

        Args:
            labels (dict): Dictionary containing 'instances'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels

    def apply_semantic(self, labels, params=None):
        """Apply transformation to semantic segmentation mask.

        Args:
            labels (dict): Dictionary containing 'semantic_mask'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels


class Compose:
    """A class for composing multiple image transformations.

    Attributes:
        transforms (list[Callable]): A list of transformation functions to be applied sequentially.

    Methods:
        __call__: Apply a series of transformations to input data.
        append: Append a new transform to the existing list of transforms.
        insert: Insert a new transform at a specified index in the list of transforms.
        __getitem__: Retrieve a specific transform or a set of transforms using indexing.
        __setitem__: Set a specific transform or a set of transforms using indexing.
        tolist: Convert the list of transforms to a standard Python list.

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """Initialize the Compose object with a list of transforms.

        Args:
            transforms (list[Callable]): A list of callable transform objects to be applied sequentially.
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Apply a series of transformations to input data.

        This method sequentially applies each transformation in the Compose object's transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the transformations in
                the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Append a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Insert a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> Compose:
        """Retrieve a specific transform or a set of transforms using indexing.

        Args:
            index (int | list[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose | Any): A new Compose object if index is a list, or a single transform if index is an int.

        Raises:
            AssertionError: If the index is not of type int or list.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # Returns the RandomPerspective transform directly
            >>> multiple_transforms = compose[[0, 1]]  # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]

    def __setitem__(self, index: list | int, value: list | int) -> None:
        """Set one or more transforms in the composition using indexing.

        Args:
            index (int | list[int]): Index or list of indices to set transforms at.
            value (Any | list[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # Replace second transform
            >>> compose[[0, 1]] = [NewTransform1(), NewTransform2()]  # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """Convert the list of transforms to a standard Python list.

        Returns:
            (list): A list containing all the transform objects in the Compose instance.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """Return a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform(BaseTransform):
    """Base class for mix transformations like Cutmix, MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the probability-based
    application of transforms and manages the mixing of multiple images and labels.

    Attributes:
        dataset (Any): The dataset object containing images and labels.
        pre_transform (Callable | None): Optional transform to apply before mixing.
        p (float): Probability of applying the mix transformation.

    Methods:
        __call__: Apply the mix transformation to the input labels.
        get_params: Prepare mixed labels and update text labels.
        get_indexes: Abstract method to get indexes of images to be mixed.
        _update_label_text: Update label text for mixed images.

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def apply_image(self, labels, params=None):
        ...         # Implement custom image mixing here
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initialize the BaseMixTransform object for mix transformations like CutMix, MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and labels for mixing.
            pre_transform (Callable | None): Optional transform to apply before mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply pre-processing transforms and cutmix/mixup/mosaic transforms to labels data.

        This method determines whether to apply the mix transform based on a probability factor. If applied, it selects
        additional images, applies pre-transforms if specified, and then performs the mix transform.

        Args:
            labels (dict[str, Any]): A dictionary containing label data for an image.

        Returns:
            (dict[str, Any]): The transformed labels dictionary, which may include mixed data from other images.

        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels

        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels.pop("mix_labels", None)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Prepare mixed labels and update text labels.

        Args:
            labels (dict[str, Any]): A dictionary containing label data for an image.

        Returns:
            (dict[str, Any]): Parameters for apply_image, apply_instances, and apply_semantic.
        """
        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic, CutMix or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        self._update_label_text(labels)
        return {"mix_labels": mix_labels}

    def get_indexes(self):
        """Get a random index for mosaic augmentation.

        Returns:
            (int): A random index from the dataset.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> index = transform.get_indexes()
            >>> print(index)  # 7
        """
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        """Update label text and class IDs for mixed labels in image augmentation.

        This method processes the 'texts' and 'cls' fields of the input labels dictionary and any mixed labels, creating
        a unified set of text labels and updating class IDs accordingly.

        Args:
            labels (dict[str, Any]): A dictionary containing label information, including 'texts' and 'cls' fields, and
                optionally a 'mix_labels' field with additional label dictionaries.

        Returns:
            (dict[str, Any]): The updated labels dictionary with unified text labels and updated class IDs.

        Examples:
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = BaseMixTransform._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels

        mix_texts = [*labels["texts"], *(item for x in labels["mix_labels"] for item in x["texts"])]
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
    augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (tuple[int, int]): Border size for height and width.

    Methods:
        get_indexes: Return a list of random indexes from the dataset.
        get_params: Compute mosaic layout parameters.
        apply_image: Allocate canvas and paste images into mosaic.
        apply_instances: Concatenate and clip instances for mosaic.
        _update_labels: Update labels with padding.
        _cat_labels: Concatenate labels and clips mosaic border instances.

    Examples:
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """Initialize the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
        augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """Return a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on the
        'buffer_enabled' attribute. It is used to choose images for creating mosaic augmentations.

        Returns:
            (list[int]): A list of random image indexes. The length of the list is n-1, where n is the number of images
                used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute mosaic layout parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary.

        Returns:
            (dict[str, Any]): Parameters including 'layout' with per-patch geometry.
        """
        params = super().get_params(labels)
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."

        s = self.imgsz
        layout = []
        if self.n == 4:
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
            for i in range(4):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                padw = x1a - x1b
                padh = y1a - y1b
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1a": x1a,
                        "y1a": y1a,
                        "x2a": x2a,
                        "y2a": y2a,
                        "x1b": x1b,
                        "y1b": y1b,
                        "x2b": x2b,
                        "y2b": y2b,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
        elif self.n == 9:
            hp, wp = -1, -1
            h0, w0 = None, None
            for i in range(9):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # center
                    c = s, s, s + w, s + h
                    h0, w0 = h, w
                elif i == 1:  # top
                    c = s, s - h, s + w, s
                elif i == 2:  # top right
                    c = s + wp, s - h, s + wp + w, s
                elif i == 3:  # right
                    c = s + w0, s, s + w0 + w, s + h
                elif i == 4:  # bottom right
                    c = s + w0, s + hp, s + w0 + w, s + hp + h
                elif i == 5:  # bottom
                    c = s + w0 - w, s + h0, s + w0, s + h0 + h
                elif i == 6:  # bottom left
                    c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
                elif i == 7:  # left
                    c = s - w, s + h0 - h, s, s + h0
                elif i == 8:  # top left
                    c = s - w, s + h0 - hp - h, s, s + h0 - hp
                padw, padh = c[:2]
                x1, y1, x2, y2 = (max(x, 0) for x in c)
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
                hp, wp = h, w
        params["layout"] = layout
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply mosaic augmentation to the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'layout'.

        Returns:
            (dict): Updated labels with mosaic image.
        """
        layout = params["layout"]
        if self.n == 4:
            img4 = np.full((self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img4
        elif self.n == 9:
            img9 = np.full((self.imgsz * 3, self.imgsz * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                img9[y1:y2, x1:x2] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply mosaic augmentation to instances.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params, including 'layout'.

        Returns:
            (dict): Updated labels with concatenated instances.
        """
        layout = params["layout"]
        mosaic_labels = []
        for item in layout:
            if self.n == 4:
                padw = item["padw"]
                padh = item["padh"]
            else:  # n == 9
                padw = item["padw"] + self.border[0]
                padh = item["padh"] + self.border[1]
            labels_patch = self._update_labels(item["labels_patch"], padw, padh, item.get("img_shape"))
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        labels.update(final_labels)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply mosaic augmentation to semantic mask.

        Args:
            labels (dict[str, Any]): Dictionary containing 'semantic_mask'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with concatenated semantic mask.
        """
        if labels.get("semantic_mask") is None and all(
            m.get("semantic_mask") is None for m in labels.get("mix_labels", [])
        ):
            return labels

        layout = params["layout"]
        if self.n == 4:
            mask4 = np.full((self.imgsz * 2, self.imgsz * 2), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask4
        elif self.n == 9:
            mask9 = np.full((self.imgsz * 3, self.imgsz * 3), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                mask9[y1:y2, x1:x2] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int, img_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Update label coordinates with padding values.

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        Args:
            labels (dict[str, Any]): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.
            img_shape (tuple[int, int] | None): Optional (h, w) of the original patch image. Needed because apply_image
                may overwrite labels["img"] with the mosaic canvas before apply_instances runs.

        Returns:
            (dict): Updated labels dictionary with adjusted instance coordinates.

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = img_shape if img_shape is not None else labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate and process labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the mosaic
        border, and removes zero-area boxes.

        Args:
            mosaic_labels (list[dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (tuple[int, int]): Original shape of the first image.
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - texts (list[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances'])
        """
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """Apply MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper [mixup: Beyond Empirical Risk
    Minimization](https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        p (float): Probability of applying MixUp augmentation.

    Methods:
        get_params: Compute MixUp parameters including blend ratio.
        apply_image: Blend images using MixUp.
        apply_instances: Concatenate instances for MixUp.

    Examples:
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        """Initialize the MixUp augmentation object.

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel values
        and labels. This implementation is designed for use with the Ultralytics YOLO framework.

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute MixUp parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary.

        Returns:
            (dict[str, Any]): Parameters including mix ratio 'r'.
        """
        params = super().get_params(labels)
        params["r"] = np.random.beta(32.0, 32.0)
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Blend images using MixUp.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'r'.

        Returns:
            (dict): Updated labels with blended image.
        """
        r = params["r"]
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Concatenate instances for MixUp.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with concatenated instances.
        """
        labels2 = labels["mix_labels"][0]
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply MixUp augmentation to semantic segmentation masks.

        Args:
            labels (dict[str, Any]): Primary image labels containing 'semantic_mask' and 'mix_labels'.
            params (dict[str, Any] | None): Parameters dict with key 'r' (mix ratio). Defaults to None.

        Returns:
            (dict[str, Any]): Updated labels with the semantic mask replaced by the mixed image's mask if r < 0.5.
        """
        if labels.get("semantic_mask") is None:
            return labels
        labels2 = labels["mix_labels"][0]
        if labels2.get("semantic_mask") is None:
            return labels
        r = params["r"]
        # Use mask from the image with higher weight to avoid fractional class indices
        if r < 0.5:
            labels["semantic_mask"] = labels2["semantic_mask"].copy()
        return labels


class CutMix(BaseMixTransform):
    """Apply CutMix augmentation to image datasets as described in the paper https://arxiv.org/abs/1905.04899.

    CutMix combines two images by replacing a random rectangular region of one image with the corresponding region from
    another image, and adjusts the labels proportionally to the area of the mixed region.

    Attributes:
        dataset (Any): The dataset to which CutMix augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before CutMix.
        p (float): Probability of applying CutMix augmentation.
        beta (float): Beta distribution parameter for sampling the mixing ratio.
        num_areas (int): Number of areas to try to cut and mix.

    Methods:
        get_params: Compute CutMix parameters including cut area and filtered indexes.
        apply_image: Copy patch from secondary image into primary image.
        apply_instances: Clip and concatenate instances for CutMix.
        _rand_bbox: Generate random bounding box coordinates for the cut region.

    Examples:
        >>> from ultralytics.data.augment import CutMix
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> cutmix = CutMix(dataset, p=0.5)
        >>> augmented_labels = cutmix(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0, beta: float = 1.0, num_areas: int = 3) -> None:
        """Initialize the CutMix augmentation object.

        Args:
            dataset (Any): The dataset to which CutMix augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply before CutMix.
            p (float): Probability of applying CutMix augmentation.
            beta (float): Beta distribution parameter for sampling the mixing ratio.
            num_areas (int): Number of areas to try to cut and mix.
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Generate random bounding box coordinates for the cut region.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            (tuple[int]): (x1, y1, x2, y2) coordinates of the bounding box.
        """
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(self.beta, self.beta)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Random center
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute CutMix parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary.

        Returns:
            (dict[str, Any]): Parameters including 'skip', 'area', and 'indexes2'.
        """
        params = super().get_params(labels)
        h, w = labels["img"].shape[:2]

        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)  # (self.num_areas, num_boxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            params["skip"] = True
            return params

        labels2 = labels["mix_labels"][0]
        area = cut_areas[np.random.choice(idx)]  # randomly select one
        ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= (0.01 if len(labels["instances"].segments) else 0.1))[0]
        if len(indexes2) == 0:
            params["skip"] = True
            return params

        params["area"] = area
        params["indexes2"] = indexes2
        params["w"] = w
        params["h"] = h
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply CutMix to the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with mixed image.
        """
        if params.get("skip"):
            return labels
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        labels2 = labels["mix_labels"][0]
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply CutMix to instances.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with mixed instances.
        """
        if params.get("skip"):
            return labels
        labels2 = labels["mix_labels"][0]
        w, h = params["w"], params["h"]
        area = params["area"]
        indexes2 = params["indexes2"]

        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)

        x1, y1, x2, y2 = area.astype(np.int32)
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)

        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"][indexes2]], axis=0)
        labels["instances"] = Instances.concatenate([labels["instances"], instances2], axis=0)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply CutMix augmentation to semantic segmentation masks.

        Args:
            labels (dict[str, Any]): Primary image labels containing 'semantic_mask' and 'mix_labels'.
            params (dict[str, Any] | None): Parameters dict with 'area' (bounding box coordinates) and 'skip' (bool
                flag). Defaults to None.

        Returns:
            (dict[str, Any]): Updated labels with the semantic mask region replaced by the mixed image's mask.
        """
        if params.get("skip"):
            return labels
        if labels.get("semantic_mask") is None:
            return labels
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        labels2 = labels["mix_labels"][0]
        if labels2.get("semantic_mask") is not None:
            mask = labels["semantic_mask"].copy()
            mask[y1:y2, x1:x2] = labels2["semantic_mask"][y1:y2, x1:x2]
            labels["semantic_mask"] = mask
        return labels


class RandomPerspective(BaseTransform):
    """Implement random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations to images and
    their associated bounding boxes, segments, and keypoints. It can be used as part of an augmentation pipeline for
    object detection and instance segmentation tasks.

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        size (tuple[int, int] | None): Output size (width, height). If None, uses the input image size.

    Methods:
        get_params: Compute affine transformation matrix and related parameters.
        apply_image: Warp the image using the affine matrix.
        apply_instances: Transform bounding boxes, segments, and keypoints.
        apply_semantic: Placeholder for semantic segmentation mask transformation.
        apply_bboxes: Transform bounding boxes using the affine matrix.
        apply_segments: Transform segments and generate new bounding boxes.
        apply_keypoints: Transform keypoints using the affine matrix.
        box_candidates: Filter transformed bounding boxes based on size and aspect ratio.

    Examples:
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float | tuple[float, float] = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        size: tuple[int, int] | None = None,
    ):
        """Initialize RandomPerspective object with transformation parameters.

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.

        Args:
            degrees (float): Degree range for random rotations.
            translate (float): Fraction of total width and height for random translation.
            scale (float | tuple[float, float]): Scaling factor interval. If float, e.g. 0.5 means resize between
                50%-150%. If tuple, interpreted as absolute (min, max) scale factors.
            shear (float): Shear intensity (angle in degrees).
            perspective (float): Perspective distortion factor.
            size (tuple[int, int] | None): Output size (width, height). If None, uses the input image size.
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.size = size

    def _compute_affine_matrix(self, img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        """Compute the affine transformation matrix without applying it.

        Args:
            img (np.ndarray): Input image used to determine center and dimensions.
            size (tuple[int, int]): Size of the output image (width, height) used for clipping translation transform.

        Returns:
            (M, scale): 3x3 transformation matrix and scale factor.
        """
        # Center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        if isinstance(self.scale, (tuple, list)):
            s = random.uniform(self.scale[0], self.scale[1])
        else:
            s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)

        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M, s

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute affine transformation parameters shared across image and instances.

        Args:
            labels (dict[str, Any]): Input labels dictionary containing 'img'.

        Returns:
            (dict): Parameters including 'M' (affine matrix), 'scale', 'orig_shape', and 'size'.
        """
        img = labels["img"]
        if (rect_shape := labels.get("rect_shape")) is not None:  # rect has higher priority
            size = (int(rect_shape[1]), int(rect_shape[0]))  # rect mode batch shape (h, w) to (w, h)
        else:
            size = (img.shape[1], img.shape[0]) if self.size is None else self.size  # w, h
        orig_shape = img.shape[:2]
        M, scale = self._compute_affine_matrix(img, size)
        return {"M": M, "scale": scale, "orig_shape": orig_shape, "size": size}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply affine warp to the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'M' and 'size'.

        Returns:
            (dict): Updated labels with warped image and 'resized_shape'.
        """
        img = labels["img"]
        M = params["M"]
        size = params["size"]
        if (size[0] != img.shape[1] or size[1] != img.shape[0]) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114))
            if img.ndim == 2:
                img = img[..., None]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply affine transformation to object instances.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params, including 'M' and 'scale'.

        Returns:
            (dict): Updated labels with transformed and filtered instances.
        """
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])

        M = params["M"]
        scale = params["scale"]

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M, params["size"])

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M, params["size"])
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*params["size"])

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        return labels

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply affine transformation to bounding boxes.

        This function applies an affine transformation to a set of bounding boxes using the provided transformation
        matrix.

        Args:
            bboxes (np.ndarray): Bounding boxes in xyxy format with shape (N, 4), where N is the number of bounding
                boxes.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).

        Returns:
            (np.ndarray): Transformed bounding boxes in xyxy format with shape (N, 4).

        Examples:
            >>> rp = RandomPerspective()
            >>> bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32)
            >>> M = np.eye(3, dtype=np.float32)
            >>> transformed_bboxes = rp.apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(
        self, segments: np.ndarray, M: np.ndarray, size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply affine transformations to segments and generate new bounding boxes.

        This function applies affine transformations to input segments and generates new bounding boxes based on the
        transformed segments. It clips the transformed segments to fit within the new bounding boxes.

        Args:
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments and M is the
                number of points in each segment.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).
            size (tuple[int, int]): Size of the output image (width, height) used for clipping the segments.

        Returns:
            bboxes (np.ndarray): New bounding boxes with shape (N, 4) in xyxy format.
            segments (np.ndarray): Transformed and clipped segments with shape (N, M, 2).

        Examples:
            >>> rp = RandomPerspective()
            >>> segments = np.random.rand(10, 500, 2)  # 10 segments with 500 points each
            >>> M = np.eye(3)  # Identity transformation matrix
            >>> new_bboxes, new_segments = rp.apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, size[0], size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Apply affine transformation to keypoints.

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation.

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, K, 3), where N is the number of instances, K is
                the number of keypoints per instance, and 3 represents (x, y, visibility).
            M (np.ndarray): 3x3 affine transformation matrix.
            size (tuple[int, int]): Size of the output image (width, height) used to determine visibility of keypoints.

        Returns:
            (np.ndarray): Transformed keypoints array with the same shape as input (N, K, 3).

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # Identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > size[0]) | (xy[:, 1] > size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply affine transformation to semantic segmentation mask.

        Args:
            labels (dict[str, Any]): Dictionary containing 'semantic_mask'.
            params (dict | None): Parameters from get_params, including 'M' and 'size'.

        Returns:
            (dict): Updated labels with transformed semantic mask.
        """
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        mask = labels["semantic_mask"]
        M = params["M"]
        size = params["size"]
        if (size[0] != mask.shape[1] or size[1] != mask.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                mask = cv2.warpPerspective(mask, M, dsize=size, flags=cv2.INTER_NEAREST, borderValue=255)
            else:
                mask = cv2.warpAffine(mask, M[:2], dsize=size, flags=cv2.INTER_NEAREST, borderValue=255)
        labels["semantic_mask"] = mask
        return labels

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        """Compute candidate boxes for further processing based on size and aspect ratio criteria.

        This method compares boxes before and after augmentation to determine if they meet specified thresholds for
        width, height, aspect ratio, and area. It's used to filter out boxes that have been overly distorted or reduced
        by the augmentation process.

        Args:
            box1 (np.ndarray): Original boxes before augmentation, shape (4, N) where N is the number of boxes. Format
                is [x1, y1, x2, y2] in absolute coordinates.
            box2 (np.ndarray): Augmented boxes after transformation, shape (4, N). Format is [x1, y1, x2, y2] in
                absolute coordinates.
            wh_thr (int): Width and height threshold in pixels. Boxes smaller than this in either dimension are
                rejected.
            ar_thr (int): Aspect ratio threshold. Boxes with an aspect ratio greater than this value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than this value are
                rejected.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            (np.ndarray): Boolean array of shape (N,) indicating which boxes are candidates. True values correspond to
                boxes that meet all criteria.

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV(BaseTransform):
    """Randomly adjust the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].

    Methods:
        apply_image: Apply random HSV augmentation to an image.

    Examples:
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> labels = augmenter(labels)
        >>> augmented_image = labels["img"]
    """

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        """Initialize the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.

        This class applies random adjustments to the HSV channels of an image within specified limits.

        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0, 1].
            sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
            vgain (float): Maximum variation for value. Should be in the range [0, 1].
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply_image(self, labels, params: dict[str, Any] | None = None):
        """Apply random HSV augmentation to an image within predefined limits.

        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value (HSV) channels. The
        adjustments are made within the limits set by hgain, sgain, and vgain during initialization.

        Args:
            labels (dict[str, Any]): A dictionary containing image data and metadata. Must include an 'img' key with the
                image as a numpy array.
            params (dict[str, Any] | None): Unused parameters for API compatibility.

        Returns:
            (dict[str, Any]): The labels dictionary with the HSV-augmented image.

        Examples:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> labels = hsv_augmenter.apply_image(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]
        if img.shape[-1] != 3:  # only apply to 3-channel (BGR) images
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype  # uint8

            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
            x = np.arange(0, 256, dtype=r.dtype)
            # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip(BaseTransform):
    """Apply a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as bounding boxes and
    keypoints.

    Attributes:
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or 'vertical'.
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.

    Methods:
        __call__: Apply the random flip transformation to an image and its annotations.

    Examples:
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p: float = 0.5, direction: str = "horizontal", flip_idx: list[int] | None = None) -> None:
        """Initialize the RandomFlip class with probability and direction.

        This class applies a random horizontal or vertical flip to an image with a given probability. It also updates
        any instances (bounding boxes, keypoints, etc.) accordingly.

        Args:
            p (float): The probability of applying the flip. Must be between 0 and 1.
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
            flip_idx (list[int] | None): Index mapping for flipping keypoints, if any.

        Raises:
            AssertionError: If direction is not 'horizontal' or 'vertical', or if p is not between 0 and 1.
        """
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute random flip parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary containing 'img' and 'instances'.

        Returns:
            (dict): Parameters including 'flip' (bool), 'h', 'w', 'direction', and 'flip_idx'.
        """
        img = labels["img"]
        instances = labels["instances"]
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        return {
            "flip": random.random() < self.p,
            "h": h,
            "w": w,
            "direction": self.direction,
            "flip_idx": self.flip_idx,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply flip to the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict): Parameters from get_params.

        Returns:
            (dict): Updated labels with flipped (or unchanged) image.
        """
        img = labels["img"]
        if params["flip"]:
            if params["direction"] == "vertical":
                img = np.flipud(img)
            elif params["direction"] == "horizontal":
                img = np.fliplr(img)
        labels["img"] = np.ascontiguousarray(img)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply flip to object instances.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances'.
            params (dict): Parameters from get_params.

        Returns:
            (dict): Updated labels with flipped (or unchanged) instances.
        """
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        if params["flip"]:
            if params["direction"] == "vertical":
                instances.flipud(params["h"])
            elif params["direction"] == "horizontal":
                instances.fliplr(params["w"])
            if params["flip_idx"] is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, params["flip_idx"], :])
        labels["instances"] = instances
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply flip to semantic segmentation mask.

        Args:
            labels (dict[str, Any]): Dictionary containing 'semantic_mask'.
            params (dict): Parameters from get_params.

        Returns:
            (dict): Updated labels with flipped (or unchanged) semantic mask.
        """
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        if params["flip"]:
            if params["direction"] == "vertical":
                labels["semantic_mask"] = np.ascontiguousarray(np.flipud(labels["semantic_mask"]))
            elif params["direction"] == "horizontal":
                labels["semantic_mask"] = np.ascontiguousarray(np.fliplr(labels["semantic_mask"]))
        return labels


class CopyPaste(BaseMixTransform):
    """CopyPaste class for applying Copy-Paste augmentation to image datasets.

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from
    different images to create new training samples.

    Attributes:
        dataset (Any): The dataset to which Copy-Paste augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before Copy-Paste.
        p (float): Probability of applying Copy-Paste augmentation.

    Methods:
        get_params: Compute CopyPaste parameters including selected instances and mask.
        apply_image: Draw contours and paste pixels for CopyPaste.
        apply_instances: Concatenate selected instances for CopyPaste.

    Examples:
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip") -> None:
        """Initialize CopyPaste object with dataset, pre_transform, and probability of applying CopyPaste."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        if mode not in ("flip", "mixup"):
            raise ValueError(f"Expected `mode` to be `flip` or `mixup`, but got {mode}.")
        self.mode = mode

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply Copy-Paste augmentation to an image and its labels."""
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            params = self.get_params(labels)
            labels = self.apply_image(labels, params)
            labels = self.apply_instances(labels, params)
            labels = self.apply_semantic(labels, params)
            return labels
        return super().__call__(labels)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute CopyPaste parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary.

        Returns:
            (dict[str, Any]): Parameters including 'instances2', 'selected', and 'im_new'.
        """
        params = {}
        if self.mode == "mixup":
            params = super().get_params(labels)
            labels2 = labels.get("mix_labels", [{}])[0]
        else:
            labels2 = {}

        h, w = labels["img"].shape[:2]
        instances = deepcopy(labels["instances"])
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        instances2 = deepcopy(labels2.get("instances")) if labels2 else None
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)

        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        selected = indexes[: round(self.p * n)]

        im_new = np.zeros((h, w), np.uint8)

        params["instances"] = instances
        params["instances2"] = instances2
        params["selected"] = selected
        params["im_new"] = im_new
        params["labels2_cls"] = labels2.get("cls")
        params["labels2_img"] = labels2.get("img")
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply CopyPaste to the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with pasted objects.
        """
        im = labels["img"].copy()

        instances2 = params["instances2"]
        selected = params["selected"]
        im_new = params["im_new"]

        for j in selected:
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, 1, cv2.FILLED)

        result = params.get("labels2_img")
        if result is None:
            result = cv2.flip(im, 1)
        if result.ndim == 2:
            result = result[..., None]

        i = im_new.astype(bool)
        im[i] = result[i]
        labels["img"] = im
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply CopyPaste to instances.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with concatenated instances.
        """
        instances = params["instances"]
        instances2 = params["instances2"]
        selected = params["selected"]
        cls = labels["cls"]
        labels2_cls = params.get("labels2_cls")

        for j in selected:
            cls = np.concatenate((cls, (labels2_cls if labels2_cls is not None else cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)

        labels["cls"] = cls
        labels["instances"] = instances
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply CopyPaste to semantic segmentation masks."""
        mask = labels.get("semantic_mask")
        if mask is None:
            return labels

        source = labels.get("mix_labels", [{}])[0].get("semantic_mask") if self.mode == "mixup" else cv2.flip(mask, 1)
        if source is None:
            return labels
        pasted = params["im_new"].astype(bool)
        mask = mask.copy()
        mask[pasted] = source[pasted]
        labels["semantic_mask"] = mask
        return labels
