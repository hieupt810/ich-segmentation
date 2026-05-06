import torchvision.transforms as transforms


def build_transform(image_size: int) -> transforms.Compose:
    """Resize to image_size and standardise to [-1, 1]"""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
