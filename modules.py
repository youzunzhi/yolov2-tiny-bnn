import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.computation import bbox_iou, bbox_wh_iou


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
    def __init__(self, module_def):
        super(RegionLoss, self).__init__()

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
        x = torch.sigmoid(prediction[..., 0])  # Center x # 8,5,H,W
        y = torch.sigmoid(prediction[..., 1])  # Center y # 8,5,H,W
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

        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        stride = 32
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
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.anchors,
                ignore_thres=self.thresh,
            )

            loss_x = self.coord_scale * nn.MSELoss()(x[obj_mask], tx[obj_mask])
            loss_y = self.coord_scale * nn.MSELoss()(y[obj_mask], ty[obj_mask])
            loss_w = self.coord_scale * nn.MSELoss()(w[obj_mask], tw[obj_mask])
            loss_h = self.coord_scale * nn.MSELoss()(h[obj_mask], th[obj_mask])
            loss_conf_obj = nn.MSELoss()(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = nn.MSELoss()(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.object_scale * loss_conf_obj + self.noobject_scale * loss_conf_noobj
            loss_cls = self.class_scale * nn.BCELoss()(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss.item(),
                "conf": loss_conf.item(),
                "cls": loss_cls.item(),
                "cls_acc": cls_acc.item(),
                "recall50": recall50.item(),
                "recall75": recall75.item(),
                "precision": precision.item(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
                "grid_size": grid_size,
            }

            return output, total_loss

    def build_targets(self, pred_boxes, pred_cls, target, anchors, ignore_thres):

        ByteTensor = torch.cuda.ByteTensor
        FloatTensor = torch.cuda.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)  # grid_size
        anchors = FloatTensor(anchors)

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        tconf = obj_mask.float() * iou_scores
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
