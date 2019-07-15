import torch
import torch.nn as nn
import torch.nn.functional as F


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        signed_tensor = tensor.sign()
        # signed_tensor[signed_tensor==0] = 1
        return signed_tensor
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = F.conv2d(input, self.weight, bias=None, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x

class RegionLoss(nn.Module):
    """Region Loss layer"""
    def __init__(self, module_def, hyper_parameters):
        super(RegionLoss, self).__init__()
        assert hyper_parameters['width']==hyper_parameters['height'], "img width must equals to height"
        self.img_size = hyper_parameters['width']

        anchors = module_def['anchors'].split(',')
        self.anchors = [float(i) for i in anchors]
        self.anchors = [(self.anchors[i], self.anchors[i + 1]) for i in range(0, len(anchors), 2)]  # list of tuple (anchor_w, anchor_h)
        self.num_classes = int(module_def['classes'])
        self.num_anchors = int(module_def['num'])

        self.object_scale = float(module_def['object_scale'])
        self.noobject_scale = float(module_def['noobject_scale'])
        self.class_scale = float(module_def['class_scale'])
        self.coord_scale = float(module_def['coord_scale'])
        self.thresh = float(module_def['thresh'])

        self.metrics = {}

    def forward(self, x, targets=None):


        FloatTensor = torch.cuda.FloatTensor
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )  # 8,5,H,W,25

        # Get outputs
        x = prediction[..., 0]  # Center x # 8,5,H,W
        y = prediction[..., 1]  # Center y # 8,5,H,W
        w = prediction[..., 2]  # Width # 8,5,H,W
        h = prediction[..., 3]  # Height # 8,5,H,W
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf # 8,5,H,W
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. # # 8,5,H,W,20

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # 8,5,H,W,4
        g = grid_size
        grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  # 1,1,H,W
        grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)  # 1,1,H,W
        anchor_w = FloatTensor(self.anchors).index_select(1, torch.LongTensor([0]).cuda()).view(1, self.num_anchors, 1, 1)
        anchor_h = FloatTensor(self.anchors).index_select(1, torch.LongTensor([1]).cuda()).view(1, self.num_anchors, 1, 1)

        pred_boxes[..., 0] = torch.sigmoid(x.data) + grid_x
        pred_boxes[..., 1] = torch.sigmoid(y.data) + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        stride = self.img_size / grid_size
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )  # 8,(H*W*5),25

        if targets is None:
            return output, 0
        else:
            assert False, "not here"