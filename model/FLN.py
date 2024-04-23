from torch import nn

from utils.transforms import *
from torch.nn.init import normal_, constant_
from .resnet import get_resnet_model
from .build_model import get_swin
from einops import rearrange
from opts.basic_ops import ConsensusModule


class FLN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet50', new_length=1, before_softmax=True, print_spec=True, consensus_type='avg', img_step=1,
                 dropout=0.8, img_feature_dim=256, crop_num=1,
                 pretrain=True, logger=None):
        super(FLN, self).__init__()
        self.num_classes = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.img_feature_dim = img_feature_dim
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.target_transforms = {86: 87, 87: 86,
                                  93: 94, 94: 93, 166: 167, 167: 166}
        self.first_layer = nn.Conv2d(3, 3, 7, stride=2, padding=3)

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
        img_step:           {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, img_step, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model, self.num_segments, logger)
        self.consensus = ConsensusModule(consensus_type)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.class_head = nn.Linear(in_features=in_features,
                                     out_features=self.num_classes)

        self.ind_head = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.Linear(in_features=512, out_features=self.num_segments)
        )
        nn.Linear(in_features=in_features,
                                   out_features=num_segments)


    def _prepare_base_model(self, base_model, num_segments, logger):
        print(('=> base model: {}'.format(base_model)))
        if 'resnet' in base_model:
            self.base_model = self._get_model(model_name=self.base_model_name, logger=logger)
            self.base_model.last_layer_name = 'fc'
            self.input_size = [336, 224]  # w * h
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _get_model(self, model_name='resnet50', logger=None):
        if 'swin' in model_name:
            model = get_swin(
                img_size=self.img_feature_dim, num_classes=self.num_classes, logger=logger)
        else:
            model = get_resnet_model(
                num_classes=self.num_classes, pretrained=self.pretrain, progress=True, model_name=self.base_model_name)
        return model

    def forward(self, input):

        # input = self.first_layer(input)

        B, S, _, _, _, = input.shape

        input = rearrange(input, "B S C H W -> (B S) C H W")

        base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = nn.Dropout(self.dropout)(base_out)
        class_out = self.class_head(base_out)
        class_out = nn.Softmax()(class_out)
        class_out = rearrange(class_out, "(B S) C -> B S C", B=B, S=S)
        output = self.consensus(class_out)
        output = torch.squeeze(output, dim=1)

        ind_out = self.ind_head(base_out)

        return output, ind_out

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
