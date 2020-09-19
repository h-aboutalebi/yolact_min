import torch
import os

class ONNX_util:

    @staticmethod
    def save_yolact(model,dataset, file_name, verbose=True):
        if(os.path.isfile(file_name) ):
            os.remove(file_name)
        dummy_input= list(range(len(dataset)))[0]
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(dummy_input)
        batch = img.unsqueeze(0)
        batch = batch.cuda()

        torch.onnx.export(model, (batch), file_name, verbose,opset_version=11)