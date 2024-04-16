from torch import nn

from utils.transforms import *
from torch.nn.init import normal_, constant_
from .resnet import get_resnet_model


class FLN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet50', new_length=1, before_softmax=True, print_spec=True,
                 dropout=0.8, img_feature_dim=256, crop_num=1,
                 pretrain=True):
        super(FLN, self).__init__()
        self.num_classes = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.target_transforms = {86: 87, 87: 86,
                                  93: 94, 94: 93, 166: 167, 167: 166}

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing FLN with base model: {}.
    FLN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model, self.num_segments)

    def _prepare_base_model(self, base_model, num_segments):
        print(('=> base model: {}'.format(base_model)))
        if 'resnet' in base_model:
            self.base_model = get_resnet_model(
                num_classes=self.num_classes, pretrained=self.pretrain, progress=True, model_name=self.base_model_name)
            self.base_model.last_layer_name = 'fc'
            self.input_size = [336, 224]  # w * h
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):

        base_out = self.base_model(input)

        base_out = nn.Softmax()(base_out)

        return base_out

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = [self.input_size[i]*256//224 for i in 
                      range(len(self.input_size))]
        return scale_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip_sth(self.target_transforms)])