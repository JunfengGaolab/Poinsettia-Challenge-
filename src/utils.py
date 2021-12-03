import os
import numpy as np
from math import sqrt
import itertools
import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor


# This function is from https://github.com/kuangliu/pytorch-ssd.
# def calc_iou_tensor(box1, box2):
#     """ Calculation of IoU based on two boxes tensor,
#         Reference to https://github.com/kuangliu/pytorch-src
#         input:
#             box1 (N, 4)  format [xmin, ymin, xmax, ymax]
#             box2 (M, 4)  format [xmin, ymin, xmax, ymax]
#         output:
#             IoU (N, M)
#     """
#     N = box1.size(0)
#     M = box2.size(0)
#
#     # (N, 4) -> (N, 1, 4) -> (N, M, 4)
#     be1 = box1.unsqueeze(1).expand(-1, M, -1)  # -1 means not changing the size of that dimension
#     # (M, 4) -> (1, M, 4) -> (N, M, 4)
#     be2 = box2.unsqueeze(0).expand(N, -1, -1)
#
#     # Left Top and Right Bottom
#     lt = torch.max(be1[:, :, :2], be2[:, :, :2])
#     rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
#
#     # compute intersection area
#     delta = rb - lt  # width and height
#     delta[delta < 0] = 0
#     # width * height
#     intersect = delta[:, :, 0] * delta[:, :, 1]
#
#     # compute bel1 area
#     delta1 = be1[:, :, 2:] - be1[:, :, :2]
#     area1 = delta1[:, :, 0] * delta1[:, :, 1]
#     # compute bel2 area
#     delta2 = be2[:, :, 2:] - be2[:, :, :2]
#     area2 = delta2[:, :, 0] * delta2[:, :, 1]
#
#     iou = intersect / (area1 + area2 - intersect)
#     return iou


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)  # default boxes number
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
        """
        ious = calc_iou_tensor(bboxes_in, self.dboxes)   # [nboxes, 8732]
        # [8732,]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)  # Find the largest IoU matched by each default box: bboxes_in
        # [nboxes,]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)  # Find the largest IoU matched by each bboxes_in default box

        # set best ious 2.0
        # Set the best default box matched by each bboxes_in as a positive sample 
        #(corresponding to the first section of the Matching strategy in the paper)
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        # Replace the matching maximum IoU bboxes_in information of the corresponding default box with best_bbox_idx
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        # Find the default box with bbox_in iou greater than 0.5, corresponding to the second item of 
        # the Matching strategy in the paper (here includes the first matched information)
        masks = best_dbox_ious > criteria
        # [8732,]
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        # Set the place where the default box is matched to the positive sample to correspond to the box information
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        # Transform format to xywh format
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Convert the box format from xywh back to ltrb, and process the predicted target score through softmax
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox

            bboxes_in: the xywh regression parameter predicted by the network
            scores_in: the predicted target probability of each default box
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        # Returns a view of the original tensor with its dimensions permuted.
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # Predicted x, y regression parameters
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # Predicted w, h regression parameters

        # Superimpose the predicted regression parameters on the default box to get the final predicted bounding box
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        # Convert the box format from xywh back to ltrb (convenient to find iou when non-maximum value suppression later) 
        #process the predicted target score through softmax
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = []
        # Traverse each image data in a batch
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))
        return outputs

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # Crop the out-of-bounds bbox
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # Remove probability information classified as background category
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        # Remove low probability targets，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []

        # Non-maximum suppression algorithm
        # scores_in (Tensor 8732 x nitems), 
        # Traverse and return each column of data, that is, the probability of the same category of 8732 targets
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            # [8732, 1] -> [8732]
            score = score.squeeze(1)

            # Exclude targets whose predicted probability is less than 0.05
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            # Sort by score from smallest to largest
            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                # Get the bboxes information of the top score_idx_sorted Tensor:[score_idx_sorted, 4]
                bboxes_sorted = bboxes[score_idx_sorted, :]
                # Get information about the first one bboxes Tensor:[4]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                # Compute the IoU of score_idx_sorted and first bboxes
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()

                # we only need iou < criteria
                # Discard all targets with the first iou> criteria (including itself)
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                # Save the index information of the first one
                candidates.append(idx)

            # Save the target information of this category after non-maximum suppression
            bboxes_out.append(bboxes[candidates, :])   # bbox Coordinate information
            scores_out.append(score[candidates])       # score information
            labels_out.extend([i] * len(candidates))   # label 

        if not bboxes_out:  
        # If it is empty, return an empty tensor, 
        #pay attention to the empty tensor size corresponding to boxes to prevent errors during verification
            return [torch.empty(size=(0, 4)), torch.empty(size=(0,), dtype=torch.int64), torch.empty(size=(0,))]

        bboxes_out = torch.cat(bboxes_out, dim=0).contiguous()
        scores_out = torch.cat(scores_out, dim=0).contiguous()
        labels_out = torch.as_tensor(labels_out, dtype=torch.long)

        # Sort the probabilities of all targets (no matter what category), take the first max_num targets
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size   # the image size of input 300
        # [38, 19, 10, 5, 3, 1]
        self.feat_size = feat_size  # The feature map size of each prediction layer

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        # [8, 16, 32, 64, 100, 300]
        self.steps = steps    # The steps of a cell on each feature layer on the original image

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales  # The scale of the default box predicted on each feature layer

        fk = fig_size / np.array(steps)     # Calculate the fk of each feature layer
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # The ratios of the default box predicted on each predicted feature layer

        self.default_boxes = []
        # size of feature and number of feature
        # Traverse each feature layer and calculate default box
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size  # scale convert to relative value[0-1]
            sk2 = scales[idx + 1] / fig_size 
            sk3 = sqrt(sk1 * sk2)
            # First add two 1:1 ratio default box width and height
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # Then add the default box width and height with different proportions to all_sizes
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # Calculate the current feature layer corresponding to all default boxes on the original image
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):  # i -> Row（y）， j -> Column（x）
                    # Calculate the center coordinates of each default box (range is between 0-1)
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        # put default_boxes to tensor type
        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)  # If the type is not changed here, an error will be reported
        self.dboxes.clamp_(min=0, max=1)  # Limit the coordinates (x, y, w, h) to between 0-1

        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        # Covert (x, y, w, h) to (xmin, ymin, xmax, ymax)
        # Convenient for subsequent calculation of IoU (when matching positive and negative samples)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]   # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]   # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]   # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]   # ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        # Return the default box of the corresponding format according to the needs
        if order == 'ltrb':
            return self.dboxes_ltrb

        if order == 'xywh':
            return self.dboxes


def dboxes300_coco():
    figsize = 300  # the image size of the network
    feat_size = [38, 19, 10, 5, 3, 1]   # the feature map size of each predictive layer
    steps = [8, 16, 32, 64, 100, 300]   # the steps of a cell on each feature layer on the original image
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]  # The scale of the default box predicted on each feature layer
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # The ratios of the default box predicted on each predicted feature layer
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    # Maximum coordinate value of all the boxes（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    # Generate a large offset for each category
    #The to here is just to make the dytpe and device that generate the tensor consistent with the boxes
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # After boxes are added with the offset of the corresponding layer, it is ensured that boxes will not overlap between different categories
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        # type: (Tensor, Tensor)
        """
            1）Get the final predicted coordinates through the predicted boxes regression parameters
            2）Convert box format from xywh back to ltrb
            3）Process the predicted target score through softmax
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox

            bboxes_in: [N, 4, 8732] is the xywh regression parameter predicted by the network
            scores_in: [N, label_num, 8732] is the predicted target probability of each default box
        """

        # Returns a view of the original tensor with its dimensions permuted.
        # [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label_num]
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # Predicted x, y regression parameters
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # Predicted w, h regression parameters

        # Superimpose the predicted regression parameters on the default box to get the final predicted bounding box
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        # scores_in: [batch, 8732, label_num]
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        # type: (Tensor, Tensor, float, int)
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # Crop the out-of-bounds bbox
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        # [num_classes] -> [8732, num_classes]
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # Remove probability information classified as background category
        bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
        scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732x20]
        labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

        # remove low scoring boxes
        # Remove low-probability targets,self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        keep = keep.nonzero().squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    def forward(self, bboxes_in, scores_in):
        # Obtain the final predicted coordinates through the predicted boxes regression parameters
        # process the predicted target score through softmax
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        # Traverse each image data in a batch
        # bboxes: [batch, 8732, 4]
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):  # split_size, split_dim
            # bbox: [1, 8732, 4]
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs
