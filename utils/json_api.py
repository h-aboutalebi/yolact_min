import json
import numpy as np
import torch
import pycocotools
from terminaltables import AsciiTable
from collections import OrderedDict
from utils.box_utils import bbox_iou, mask_iou
from utils import timer
from utils.output_utils import after_nms
from data.config import cfg, COCO_LABEL_MAP

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
cuda = torch.cuda.is_available()

class Make_json:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []
        self.coco_cats = {}

        for coco_id, real_id in COCO_LABEL_MAP.items():
            class_id = real_id - 1
            self.coco_cats[class_id] = coco_id

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({'image_id': int(image_id),
                               'category_id': self.coco_cats[int(category_id)],
                               'bbox': bbox,
                               'score': float(score)})

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({'image_id': int(image_id),
                               'category_id': self.coco_cats[int(category_id)],
                               'segmentation': rle,
                               'score': float(score)})

    def dump(self):
        dump_arguments = [(self.bbox_data, f'results/bbox_detections.json'),
                          (self.mask_data, f'results/mask_detections.json')]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)


class APDataObject:
    """Stores all the information necessary to calculate the AP for one IoU and one class."""

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def prep_metrics(ap_data, nms_outs, gt, gt_masks, h, w, num_crowd, image_id, make_json, cocoapi):
    """ Returns a list of APs for this image, with each element being for a class  """

    with timer.env('After NMS'):
        pred_classes, pred_confs, pred_boxes, pred_masks = after_nms(nms_outs, h, w)
        if pred_classes.size(0) == 0:
            return

        pred_classes = list(pred_classes.cpu().numpy().astype(int))
        pred_confs = list(pred_confs.cpu().numpy().astype(float))
        pred_masks = pred_masks.view(-1, h * w).cuda() if cuda else pred_masks.view(-1, h * w)
        pred_boxes = pred_boxes.cuda() if cuda else pred_boxes

    if cocoapi:
        with timer.env('Output json'):
            pred_boxes = pred_boxes.cpu().numpy()
            pred_masks = pred_masks.view(-1, h, w).cpu().numpy()

            for i in range(pred_masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (pred_boxes[i, 3] - pred_boxes[i, 1]) * (pred_boxes[i, 2] - pred_boxes[i, 0]) > 0:
                    make_json.add_bbox(image_id, pred_classes[i], pred_boxes[i, :], pred_confs[i])
                    make_json.add_mask(image_id, pred_classes[i], pred_masks[i, :, :], pred_confs[i])
        return

    with timer.env('Prepare gt'):
        gt_boxes = torch.Tensor(gt[:, :4])
        gt_boxes[:, [0, 2]] *= w
        gt_boxes[:, [1, 3]] *= h
        gt_classes = list(gt[:, 4].astype(int))
        gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

        if num_crowd > 0:
            split = lambda x: (x[-num_crowd:], x[:-num_crowd])
            crowd_boxes, gt_boxes = split(gt_boxes)
            crowd_masks, gt_masks = split(gt_masks)
            crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Eval Setup'):
        mask_iou_cache = mask_iou(pred_masks, gt_masks)
        bbox_iou_cache = bbox_iou(pred_boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = mask_iou(pred_masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = bbox_iou(pred_boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        iou_types = [('box', lambda i, j: bbox_iou_cache[i, j].item(), lambda i, j: crowd_bbox_iou_cache[i, j].item()),
                     ('mask', lambda i, j: mask_iou_cache[i, j].item(), lambda i, j: crowd_mask_iou_cache[i, j].item())]

    timer.start('Main loop')
    for _class in set(pred_classes + gt_classes):
        num_gt_per_class = gt_classes.count(_class)

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func in iou_types:
                gt_used = [False] * len(gt_classes)
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_per_class)

                for i, pred_class in enumerate(pred_classes):
                    if pred_class != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j, gt_class in enumerate(gt_classes):
                        if gt_used[j] or gt_class != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(pred_confs[i], True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(pred_confs[i], False)
    timer.stop('Main loop')


def calc_map(ap_data):
    print('\nCalculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    row1 = list(all_maps['box'].keys())
    row1.insert(0, ' ')

    row2 = list(all_maps['box'].values())
    row2 = [round(aa, 2) for aa in row2]
    row2.insert(0, 'box')

    row3 = list(all_maps['mask'].values())
    row3 = [round(aa, 2) for aa in row3]
    row3.insert(0, 'mask')

    table = [row1, row2, row3]
    table = AsciiTable(table)
    return table.table, row2, row3

