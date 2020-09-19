import torch
import onnx
import argparse
import torch.backends.cudnn as cudnn

from data.coco import COCODetection
from modules.build_yolact import Yolact
from utils.augmentations import BaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from utils.json_api import APDataObject, Make_json, prep_metrics, calc_map
from utils.onnx_util import ONNX_util
from utils.output_utils import  NMS
from data.config import cfg, update_config

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--trained_model', default='yolact_base_54_800000.pth', type=str)
parser.add_argument('--visual_top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--max_num', default=-1, type=int, help='The maximum number of images for test, set to -1 for all.')
parser.add_argument('--cocoapi', action='store_true', help='Whether to use cocoapi to evaluate results.')
parser.add_argument('--onnx', action='store_false', help='Whether to do onnx model generation')



def evaluate(net, dataset, max_num=-1, during_training=False, cocoapi=False, traditional_nms=False):
    frame_times = MovingAverage()
    dataset_size = len(dataset) if max_num < 0 else min(max_num, len(dataset))
    dataset_indices = list(range(len(dataset)))
    dataset_indices = dataset_indices[:dataset_size]
    progress_bar = ProgressBar(40, dataset_size)

    # For each class and iou, stores tuples (score, isPositive)
    # Index ap_data[type][iouIdx][classIdx]
    ap_data = {'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
               'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]}
    make_json = Make_json()

    for i, image_idx in enumerate(dataset_indices):
        timer.reset()

        with timer.env('Data loading'):
            img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

            batch = img.unsqueeze(0)
            if cuda:
                batch = batch.cuda()

        with timer.env('Network forward'):
            #changed
            net_outs = net(batch)
            nms_outs = NMS(net_outs, traditional_nms)
            prep_metrics(ap_data, nms_outs, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], make_json, cocoapi)

        # First couple of images take longer because we're constructing the graph.
        # Since that's technically initialization, don't include those in the FPS calculations.
        fps = 0
        if i > 1 and not during_training:
            frame_times.add(timer.total_time())
            fps = 1 / frame_times.get_avg()

        progress = (i + 1) / dataset_size * 100
        progress_bar.set_val(i + 1)
        print('\rProcessing:  %s  %d / %d (%.2f%%)  %.2f fps  ' % (
            repr(progress_bar), i + 1, dataset_size, progress, fps), end='')
    else:
        table, box_row, mask_row = calc_map(ap_data)
        print(table)
        return table, box_row, mask_row


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
cuda = torch.cuda.is_available()

if __name__ == '__main__':
    args = parser.parse_args()
    strs = args.trained_model.split('_')
    config = f'{strs[-3]}_{strs[-2]}_config'

    update_config(config)
    print(f'\nUsing \'{config}\' according to the trained_model.\n')

    with torch.no_grad():
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info, augmentation=BaseTransform())

        net = Yolact()
        # net=onnx.load("yolact.onnx")
        # net= torch.jit.script(net)
        net.load_weights('weights/' + args.trained_model, cuda)
        net.eval()
        if(args.onnx):
            ONNX_util.save_yolact(net,dataset, "yolact.onnx")
        print('\nModel loaded.\n')

        if cuda:
            net = net.cuda()

        evaluate(net, dataset, args.max_num, False, args.cocoapi, args.traditional_nms)
