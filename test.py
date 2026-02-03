from quaternion.qwnet import QFNet
from utils.Evaluator import Evaluator
from utils.img_read_save import img_save, image_read_cv2
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/QFNetv4_parameter_Medical_best_11-05-10-15.pth"
for dataset_name in ["MIF"]:
    print("\n"*2+"="*80)
    model_name="QFNetv3"
    print("The test result of " + dataset_name + ' :')
    test_folder=os.path.join('test_img', dataset_name) 
    test_out_folder=os.path.join('test_result', dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QFNet().to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "MRI")):
            data_IR = image_read_cv2(os.path.join(test_folder, "MRI", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_Y, data_VIS_Cr, data_VIS_Cb = cv2.split(image_read_cv2(os.path.join(test_folder, "Others", img_name), mode='YCrCb'))
            data_VIS_Y = data_VIS_Y[np.newaxis, np.newaxis, ...] / 255.0

            data_IR, data_VIS_Y = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS_Y)
            data_VIS_Y, data_IR = data_VIS_Y.cuda(), data_IR.cuda()

            data_Fuse = model(data_VIS_Y, data_IR)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            # img_save(fi, img_name.split(sep='.')[0], test_out_folder)
            # float32 to uint8
            fi = fi.astype(np.uint8)
            # concatnate
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2BGR)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)



    eval_folder=test_out_folder  
    ori_img_folder=test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder, "MRI")):
        ir = image_read_cv2(os.path.join(ori_img_folder, "MRI", img_name), mode='GRAY')
        vis_YCrCb = image_read_cv2(os.path.join(ori_img_folder, "Others", img_name), mode='YCrCb')
        vi = cv2.split(vis_YCrCb)[0]  # Extract Y channel
        fi = cv2.split(image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), mode='YCrCb'))[0]  # Extract Y channel

        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                   , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                   , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                   , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
          + str(np.round(metric_result[1], 2)) + '\t'
          + str(np.round(metric_result[2], 2)) + '\t'
          + str(np.round(metric_result[3], 2)) + '\t'
          + str(np.round(metric_result[4], 2)) + '\t'
          + str(np.round(metric_result[5], 2)) + '\t'
          + str(np.round(metric_result[6], 2)) + '\t'
          + str(np.round(metric_result[7], 2))
          )
    print("=" * 80)