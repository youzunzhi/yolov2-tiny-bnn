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

        # self.object_scale = float(module_def['object_scale'])
        # self.object_scale = 10
        self.noobject_scale = float(module_def['noobject_scale'])
        self.class_scale = float(module_def['class_scale'])
        self.coord_scale = float(module_def['coord_scale'])
        self.thresh = float(module_def['thresh'])
        self.rescore = int(module_def['rescore'])

        self.metrics = {}

    def forward(self, x, seen, targets=None):

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

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) / grid_size,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )  # 8,(H*W*5),25

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, coord_mask_scale, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                targets=targets,
                anchors=self.anchors,
                ignore_thresh=self.thresh,
                seen=seen
            )

            coord_mask = coord_mask_scale > 0
            # loss_x = ((x[coord_mask]-tx[coord_mask])**2*coord_mask_scale[coord_mask]).sum()  # scaled SSELoss
            # loss_y = ((y[coord_mask]-ty[coord_mask])**2*coord_mask_scale[coord_mask]).sum()  # scaled SSELoss
            # loss_w = ((w[coord_mask]-tw[coord_mask])**2*coord_mask_scale[coord_mask]).sum()  # scaled SSELoss
            # loss_h = ((h[coord_mask]-th[coord_mask])**2*coord_mask_scale[coord_mask]).sum()  # scaled SSELoss
            loss_x = nn.MSELoss(reduction='sum')(x * coord_mask_scale**0.5, tx * coord_mask_scale**0.5)
            loss_y = nn.MSELoss(reduction='sum')(y * coord_mask_scale**0.5, ty * coord_mask_scale**0.5)
            loss_w = nn.MSELoss(reduction='sum')(w * coord_mask_scale**0.5, tw * coord_mask_scale**0.5)
            loss_h = nn.MSELoss(reduction='sum')(h * coord_mask_scale**0.5, th * coord_mask_scale**0.5)
            loss_coord = loss_x + loss_y + loss_w + loss_h
            self.object_scale = noobj_mask.sum() / obj_mask.sum()
            loss_conf_obj = self.object_scale * nn.MSELoss(reduction='sum')(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.noobject_scale * nn.MSELoss(reduction='sum')(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_cls = self.class_scale * nn.BCELoss(reduction='sum')(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_coord + loss_conf_obj + loss_conf_noobj + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            # conf50 = (pred_conf > 0.5).float()
            # iou50 = (iou_scores > 0.5).float()
            # iou75 = (iou_scores > 0.75).float()
            # detected_mask = conf50 * class_mask * tconf
            # precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            # recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            # recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss.item(),
                "loss_coord": loss_coord.item(),
                "loss_conf_obj": loss_conf_obj.item(),
                "loss_conf_noobj": loss_conf_noobj.item(),
                "loss_cls": loss_cls.item(),
                "avg_iou": iou_scores[obj_mask].mean(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
                "cls_acc": cls_acc.item(),
                # "recall50": recall50.item(),
                # "recall75": recall75.item(),
                # "precision": precision.item(),
                "grid_size": grid_size,
            }

            return output, total_loss

    def build_targets(self, pred_boxes, pred_cls, targets, anchors, ignore_thresh, seen):

        ByteTensor = torch.cuda.ByteTensor
        FloatTensor = torch.cuda.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)  # grid_size
        anchors = FloatTensor(anchors)

        # Output tensors
        coord_mask_scale = FloatTensor(nB, nA, nG, nG).fill_(0)
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # if iter < 12800, learn anchor box
        if seen < 12800:
            tx.fill_(0.5)
            ty.fill_(0.5)
            coord_mask_scale.fill_(1)

        # Convert to position relative to box
        target_boxes = targets[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = targets[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        coord_mask_scale[b, best_n, gj, gi] = (2 - targets[:,4] * targets[:,5]) * self.coord_scale
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # # Set noobj mask to zero where iou exceeds ignore threshold
        # for i, anchor_ious in enumerate(ious.t()):
        #     noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Set noobj mask to zero where iou of pred_box and any target box exceeds ignore threshold
        for target_box in target_boxes:
            target_box_repeat = target_box.repeat(nB, nA, nG, nG, 1)
            pred_ious = bbox_iou(pred_boxes, target_box_repeat, x1y1x2y2=False)
            noobj_mask[pred_ious>ignore_thresh] = 0

        # Target Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Target Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0])
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1])
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        if self.rescore:
            tconf = obj_mask.float() * iou_scores # rescore
        else:
            tconf = obj_mask.float()
        return iou_scores, class_mask, coord_mask_scale, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

class RegionLoss2(nn.Module):
    """Region Loss layer"""
    def __init__(self, module_def):
        super(RegionLoss2, self).__init__()

        anchors = module_def['anchors'].split(',')
        self.anchors = [float(i) for i in anchors]
        self.anchors = [(self.anchors[i], self.anchors[i + 1]) for i in range(0, len(anchors), 2)]  # list of tuple (anchor_w, anchor_h)
        self.num_classes = int(module_def['classes'])
        self.num_anchors = int(module_def['num'])

        self.object_scale = float(module_def['object_scale'])
        # self.object_scale = 10
        self.noobject_scale = float(module_def['noobject_scale'])
        self.class_scale = float(module_def['class_scale'])
        self.coord_scale = float(module_def['coord_scale'])
        self.thresh = float(module_def['thresh'])
        self.rescore = int(module_def['rescore'])

        self.metrics = {}


    def forward(self, output, seen, target):
        # output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB, nA, (5 + nC), nH, nW)
        x = F.sigmoid(output.index_select(2, (torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y = F.sigmoid(output.index_select(2, (torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w = output.index_select(2, (torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h = output.index_select(2, (torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, (torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls = output.index_select(2, (torch.linspace(5, 5 + nC - 1, nC).long().cuda()))
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(
            nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1,
                                                                                      torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1,
                                                                                      torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        pred_boxes[0] = torch.reshape(x.data, (1, nB * nA * nH * nW)) + grid_x
        pred_boxes[1] = torch.reshape(y.data, (1, nB * nA * nH * nW)) + grid_y
        pred_boxes[2] = torch.reshape(torch.exp(w.data), (1, nB * nA * nH * nW)) * anchor_w
        pred_boxes[3] = torch.reshape(torch.exp(h.data), (1, nB * nA * nH * nW)) * anchor_h
        pred_boxes = torch.FloatTensor(pred_boxes.transpose(0, 1).contiguous().view(-1, 4).size()).copy_(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(pred_boxes,
                                                                                                    target.data,
                                                                                                    self.anchors,
                                                                                                    nA, nC, nH, nW,
                                                                                                    self.noobject_scale,
                                                                                                    self.object_scale,
                                                                                                    self.thresh, seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data.item())

        tx = (tx.cuda())
        ty = (ty.cuda())
        tw = (tw.cuda())
        th = (th.cuda())
        tconf = (tconf.cuda())
        tcls = (tcls[cls_mask].long().cuda())

        coord_mask = (coord_mask.cuda())
        conf_mask = (conf_mask.cuda().sqrt())
        cls_mask = (cls_mask.view(-1, 1).repeat(1, nC).cuda())
        cls = cls[cls_mask].view(-1, nC)


        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / 2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        self.seen, nGT, nCorrect, nProposals, loss_x.data.item(), loss_y.data.item(), loss_w.data.item(),
        loss_h.data.item(), loss_conf.data.item(), loss_cls.data.item(), loss.data.item()))

        return 0, loss

    def build_targets(self, pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale,
                      sil_thresh, seen):
        nB = target.size(0)
        nA = num_anchors
        nC = num_classes
        anchor_step = len(anchors) // num_anchors
        conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tx = torch.zeros(nB, nA, nH, nW)
        ty = torch.zeros(nB, nA, nH, nW)
        tw = torch.zeros(nB, nA, nH, nW)
        th = torch.zeros(nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)

        nAnchors = nA * nH * nW
        nPixels = nH * nW
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            for t in range(50):
                if target[b][t * 5 + 1] == 0:
                    break
                gx = target[b][t * 5 + 1] * nW
                gy = target[b][t * 5 + 2] * nH
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            temp_thresh = cur_ious > sil_thresh
            conf_mask[b][temp_thresh.view(conf_mask[b].shape)] = 0
        if seen < 12800:
            if anchor_step == 4:
                tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1, nA,
                                                                                                                  1,
                                                                                                                  1).repeat(
                    nB, 1, nH, nW)
                ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1,
                                                                                            torch.LongTensor([2])).view(
                    1, nA, 1, 1).repeat(nB, 1, nH, nW)
            else:
                tx.fill_(0.5)
                ty.fill_(0.5)
            tw.zero_()
            th.zero_()
            coord_mask.fill_(1)

        nGT = 0
        nCorrect = 0
        for b in range(nB):
            for t in range(50):
                if target[b][t * 5 + 1] == 0:
                    break
                nGT = nGT + 1
                best_iou = 0.0
                best_n = -1
                min_dist = 10000
                gx = target[b][t * 5 + 1] * nW
                gy = target[b][t * 5 + 2] * nH
                gi = int(gx)
                gj = int(gy)
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                gt_box = [0, 0, gw, gh]
                for n in range(nA):
                    aw = anchors[anchor_step * n]
                    ah = anchors[anchor_step * n + 1]
                    anchor_box = [0, 0, aw, ah]
                    iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                    if anchor_step == 4:
                        ax = anchors[anchor_step * n + 2]
                        ay = anchors[anchor_step * n + 3]
                        dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                    elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                        best_iou = iou
                        best_n = n
                        min_dist = dist

                gt_box = [gx, gy, gw, gh]
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = object_scale
                tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
                ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
                tw[b][best_n][gj][gi] = torch.log(gw / anchors[anchor_step * best_n])
                th[b][best_n][gj][gi] = torch.log(gh / anchors[anchor_step * best_n + 1])
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
                tconf[b][best_n][gj][gi] = iou
                tcls[b][best_n][gj][gi] = target[b][t * 5]
                if iou > 0.5:
                    nCorrect = nCorrect + 1

        return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls
