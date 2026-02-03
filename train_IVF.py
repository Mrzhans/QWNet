# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
from quaternion.qwnet import QWNet
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss_vif import fusion_loss_vif

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

def main():
    model_str = 'QFNetv4'

    # . Set the hyper-parameters for training
    num_epochs = 10000

    lr = 1e-5
    weight_decay = 0
    batch_size = 64

    clip_grad_norm_value = 0.01
    optim_step = 30
    optim_gamma = 0.5

    # Initialize the variable to track the improvement of min_loss
    no_improvement_count = 0
    patience = 1000 # 20 epochs without improvement to stop

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QWNet().to(device=device)

    # optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

    # G_loss, i_loss, ssim_loss
    loss_vif = fusion_loss_vif()


    # data loader
    trainloader = DataLoader(H5Dataset(r"data/Fusion_train_imgsize_128_stride_200.h5"),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    # trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          num_workers=4)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

    '''
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    '''

    step = 0
    torch.backends.cudnn.benchmark = True
    prev_time = time.time()

    # Initialize the minimum loss to a very large value
    min_loss = float('inf')
    best_model_path = os.path.join("models", f"QFNetv4_channel_8_16_32_64_128_{timestamp}.pth")

    for epoch in range(num_epochs):
        train_tqdm = tqdm(trainloader, total=len(trainloader))
        ''' train '''
        for i, (data_VIS, data_IR) in enumerate(train_tqdm):
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
            model.train()

            model.zero_grad()

            optimizer.zero_grad()
            
            fuse_image = model(data_VIS, data_IR)
            # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
            fused_image = torch.clamp(fuse_image, min=0, max=1)
            fusion_loss, loss_gradient, loss_l1, loss_SSIM = loss_vif(image_A=data_VIS, image_B=data_IR, image_fused=fuse_image)
            fusion_loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            train_tqdm.set_postfix(epoch=epoch, loss_gradient=loss_gradient.item(), loss_l1=loss_l1.item(),
                                        loss_SSIM=loss_SSIM.item(), total_loss=fusion_loss.item(),)

        # Check if the current total_loss is the lowest
        if fusion_loss.item() < min_loss:
            min_loss = fusion_loss.item()
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with total_loss: {min_loss}")
            # Reset the counter when there is an improvement
            no_improvement_count = 0
        else:
            # Increment the counter if there is no improvement
            no_improvement_count += 1

        # adjust the learning rate
        scheduler.step()
        
        # Check the counter value to decide whether to stop training
        if no_improvement_count >= patience:
            print(f"No improvement in {patience} epochs. Stopping training.")
            break  

        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6
        
    # Save the final model at the end of all epochs
    torch.save(model.state_dict(), os.path.join("models", f"QFNetv4_channel_8_16_32_64_128_{timestamp}.pth"))

if __name__ == '__main__':
    # windows running
    torch.multiprocessing.freeze_support()
    main()
