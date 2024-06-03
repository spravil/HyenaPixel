import hyenapixel.models
import timm
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class TimmModel(BaseModule):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        self.model = timm.create_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
