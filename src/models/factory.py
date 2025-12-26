from torchvision import models


MODELS_REGISTRY = {
    "resnet18": models.resnet18,

}


def build_model(model_config):
    pass
