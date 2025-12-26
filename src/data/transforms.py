from torchvision import transforms


def get_transforms(config):
    img_size = config.data.image_size
    
    TRANSFORMS = {
        "basic": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]),
        "simple": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ]),
    }
    
    train_transform = TRANSFORMS[config.data.transform]
    val_transform = TRANSFORMS["basic"]

    class_transforms_dict = {}
    if hasattr(config.data, "class_transforms_map"):
        for cls, t_name in config.data.class_transforms_map.items():
            class_transforms_dict[cls] = TRANSFORMS[t_name]

    return train_transform, val_transform, class_transforms_dict
