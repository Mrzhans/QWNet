import os
import h5py
import numpy as np
from utils.img_read_save import image_read_cv2
from tqdm import tqdm
from utils.bgr_and_ycrcb import bgr_to_ycrcb

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

data_name="MSRS_train"
img_size=128   #patch size
stride=200     #patch stride

IR_files = sorted(get_img_file(r"/data2/yjt/Datasets/MSRS/train/ir"))
VIS_files   = sorted(get_img_file(r"/data2/yjt/Datasets/MSRS/train/vis"))

assert len(IR_files) == len(VIS_files)
h5f = h5py.File(os.path.join('data/',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'), 
                    'w')
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
train_num=0
for i in tqdm(range(len(IR_files))):
        I_VIS, CBCR = bgr_to_ycrcb(VIS_files[i])  # [3, H, W] Uint8->float32
        I_VIS = np.expand_dims(I_VIS, axis=0) / 255
        # 红外光一直都是单通道图片
        I_IR = image_read_cv2(IR_files[i], 'GRAY').astype(np.float32)[None, :, :] / 255.  # [1, H, W] Float32
        # crop    
        I_IR_Patch_Group = Im2Patch(I_IR, img_size, stride)
        I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  # (3, 256, 256, 12)
        
        for ii in range(I_IR_Patch_Group.shape[-1]):
            bad_IR = is_low_contrast(I_IR_Patch_Group[0,:,:,ii])
            bad_VIS = is_low_contrast(I_VIS_Patch_Group[0,:,:,ii])
            # Determine if the contrast is low
            if not (bad_IR or bad_VIS):
                avl_IR= I_IR_Patch_Group[:,:,:,ii]  #  available IR
                avl_VIS= I_VIS_Patch_Group[:,:,:,ii]
                # avl_IR=avl_IR[None,...]
                # avl_VIS=avl_VIS[None,...]
                
                h5_ir.create_dataset(str(train_num),     data=avl_IR, 
	                            dtype=avl_IR.dtype,   shape=avl_IR.shape)
                h5_vis.create_dataset(str(train_num),    data=avl_VIS, 
	                            dtype=avl_VIS.dtype,  shape=avl_VIS.shape)
                train_num += 1        

h5f.close()

with h5py.File(os.path.join('data',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'), "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
    

    



    
