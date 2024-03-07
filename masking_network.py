from torch import nn
import pytorch_lightning as pl
import torch
from unet import Unet
from prodigyopt import Prodigy
from kornia.filters import  filter2d
import torch.nn.functional as F
from model_extractors import *

def cosine_mean(x1,x2 ):
    return  (1 - torch.mean(F.cosine_similarity(x1, x2, dim=-1)))/2 # 0 is close 1 is far


class EQ_dist(torch.nn.Module):
    def __init__(self):
        super(EQ_dist, self).__init__()

    def forward(self, target, pred_mask,pred):
        pred = F.softmax(pred, dim=-1)
        pred_mask = F.softmax(pred_mask, dim=-1)
        if  len(pred.shape) == len(target.shape):
            target_max_index =target.argmax(dim=-1)
        else :
            target_max_index = target
        pred = pred[:,target_max_index]
        pred_mask = pred_mask[:,target_max_index]
        return  (pred_mask - pred).pow(2).mean()
        return  (pred_mask - 1).pow(2).mean()





class Masking_Loss(nn.Module):
    def __init__(self,constrastive = False, supervised = False,inversed = False):
        super(Masking_Loss, self, ).__init__()
        self.constrastive = constrastive
        self.supervised = supervised
        self.measure_emb = cosine_mean
        self.n_cor = 1e-6
        self.inverse = inversed

        self.measure_output = EQ_dist()

    def unsupervised_loss(self, masks, true_vector_emb, masked_vector_emb, neg_masked_vector_emb):
        masking_ratio = torch.mean(torch.abs(masks))
        pdist_emb = self.measure_emb(true_vector_emb, masked_vector_emb)
        if self.inverse:
            pdist_emb = 1 - pdist_emb
            masking_ratio = 1 - masking_ratio
        if self.constrastive:
            ndist_emb = self.measure_emb(masked_vector_emb,neg_masked_vector_emb)
            loss = 1/2 * masking_ratio + 4*  pdist_emb  - ndist_emb  * (pdist_emb + self.n_cor) + self.n_cor
        else: 
            loss = 4* pdist_emb + 1/2 *  masking_ratio
        return   loss, pdist_emb, masking_ratio
    
    def supervised_loss(self, masks, 
                true_vector_emb, true_vector_output,
                masked_vector_emb, masked_vector_output, 
                neg_masked_vector_emb, neg_masked_vector_output, target = None):
        masking_ratio = torch.mean(torch.abs(masks))
        pdist_emb = self.measure_emb(true_vector_emb, masked_vector_emb)
        if target is not None:
            pdist_out = self.measure_output(target, masked_vector_output, true_vector_output)
        else:
            pdist_out = self.measure_output(true_vector_output, masked_vector_output, true_vector_output)

        if self.inverse:
            pdist_out, masking_ratio  = 1 - pdist_out, 1 - masking_ratio  
        
        if self.constrastive:
            pndist_out = self.measure_output(target,neg_masked_vector_output, true_vector_output)
            
            ndist_emb = self.measure_emb(masked_vector_emb,neg_masked_vector_emb)
            ndist_out = self.measure_output(target, masked_vector_output,neg_masked_vector_output)
            if self.inverse:
                ndist_emb, ndist_out = 1 - ndist_emb, 1 - ndist_out
                pndist_out, pdist_emb = 1 - pndist_out, 1 - pdist_emb
        
   
                
        loss =  masking_ratio  + 2 * pdist_out 
        if self.constrastive:
            loss = masking_ratio  + pdist_out   - pndist_out 
        return loss, pdist_out, masking_ratio
    def forward(self, masks, 
                true_vector_emb, true_vector_output,
                masked_vector_emb, masked_vector_output, 
                neg_masked_vector_emb, neg_masked_vector_output, target = None):
        

        if  self.supervised:
            return self.supervised_loss(masks, true_vector_emb,
                                         true_vector_output, masked_vector_emb, 
                                         masked_vector_output, neg_masked_vector_emb, 
                                         neg_masked_vector_output, target = target)
        return self.unsupervised_loss(masks, true_vector_emb, masked_vector_emb, neg_masked_vector_emb)

    
class masking_network(pl.LightningModule):
    def __init__(self, epochs, batch_size,
                   lr = 1,img_size = (224,224), 
                   partition = 1, noise_mask = False, blur = False, constrastive=False, variational = False , supervised = False, inverse = False):
        super().__init__()
        self.save_hyperparameters()
        self.noise_mask = noise_mask
        self.negative = constrastive
        self.patcher = partition is not None
        self.blur = blur
        self.partition = partition
        self.supervised = supervised
        self.variational = variational    
        self.loss_func       = Masking_Loss(
            constrastive=constrastive,
            supervised=supervised,
            inversed=inverse
            )
        self.learning_rate    =   lr 
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        if self.patcher:
            self.patching_layer = nn.Conv2d(1, 2 if self.variational else 1, kernel_size=partition, stride=partition, padding=0)
        self.masking_network = None
        self.frozen_network  = None
        self.global_pool = None
        print(f"constrastive {constrastive}")
        
    def gen_mask(self,x,get_normal = False):
        mask =  self.masking_network(x)
        if self.patcher:
            mask = self.patching_layer(mask)   
            mask =  torch.sigmoid(mask)
            
        if self.partition > 1:
                mask = F.interpolate(mask, size=self.img_size
                                     ,mode="bilinear"
                                     )
        if self.supervised:
            rand_kernel = torch.rand(size=(self.batch_size,21,21))
            mask = filter2d(mask, rand_kernel, normalized=True,border_type='circular')
    
        
        if self.noise_mask:
            device = next(self.masking_network.parameters()).device
            noise = torch.normal(0,1,size=x.shape).to( device)
            data_mean = torch.tensor(self.data_mean, device =  device)[None, :, None, None]
            data_std = torch.tensor(self.data_std, device =  device)[None, :, None, None]
            noise = noise*data_std+data_mean
            x_masked =  x * (mask.clamp(0,1)) + noise * (1 - mask.clamp(0,1)) 
            if self.negative:
                self.negative_img =  x * (1 - mask.clamp(0,1)) + noise * (mask.clamp(0,1))
                  
        else:
            x_masked =  x * (mask.clamp(0,1))
            if self.negative:
                self.negative_img = x * (1 - mask.clamp(0,1))

        return mask, x_masked
    
    def gen_representations(self,x, x_masked):
        pred_emb =  self.frozen_network.get_embeddings(x)
        pred_masked_emb =  self.frozen_network.get_embeddings(x_masked)
        pred_neg_emb = self.frozen_network.get_embeddings(self.negative_img) if self.negative else None
        if self.supervised:
            pred = self.frozen_network.get_output_from_embeddings(pred_emb)        
            pred_neg = self.frozen_network.get_output_from_embeddings(pred_neg_emb) if self.negative else None
            pred_masked = self.frozen_network.get_output_from_embeddings(pred_masked_emb)
        else:
            pred, pred_neg, pred_masked = None, None, None

        return pred_emb, pred, pred_masked_emb, pred_masked , pred_neg_emb, pred_neg
    
    def run_step(self,x,target = None):
        mask, x_masked = self.gen_mask(x)
        pred_emb, pred, pred_masked_emb, pred_masked,pred_neg_emb,  pred_neg = self.gen_representations(x=x,x_masked=x_masked)
        
        loss, pdist, masking_ratio  = self.loss_func(
            mask, pred_emb, pred, pred_masked_emb, pred_masked, pred_neg_emb, pred_neg, target = target)
        return loss, pdist, masking_ratio, pred, pred_masked 

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch

        if self.masking_network.freeze_backbone:
            self.masking_network.encoder.eval()
        loss, pdist, masking_ratio, _, _ = self.run_step(x,target= y if self.supervised else None)
        self.log("train_loss", loss, batch_size=self.batch_size)
        self.log("train_mask_norm",masking_ratio, batch_size=self.batch_size)
        self.log("train_dist",pdist, batch_size=self.batch_size)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        loss, pdist, masking_ratio, _ , _ = self.run_step(x,target= y if self.supervised else None)
        self.log("val_loss", loss, batch_size=self.batch_size)
        self.log("val_mask_norm",masking_ratio, batch_size=self.batch_size)
        self.log("val_dist",pdist, batch_size=self.batch_size)

        return loss
    
    def test_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        loss, pdist, masking_ratio, _ , _ = self.run_step(x,target= y if self.supervised else None)
        self.log("test_loss", loss, batch_size=self.batch_size)
        self.log("test_mask_norm",masking_ratio, batch_size=self.batch_size)
        self.log("test_dist",pdist, batch_size=self.batch_size)
        return loss
    
    def forward(self, x):
        return self.gen_mask(x=x)
    
    def configure_optimizers(self):
        optim_spec = lambda params: Prodigy(
            params, lr=1, weight_decay=0.01, safeguard_warmup=True)
        # Only optimize decoder if backbone is frozen
        if self.masking_network.freeze_backbone:
            if self.patcher:
                optimizer =  optim_spec(
                    list(self.masking_network.decoder.parameters()) + 
                    list(self.patching_layer.parameters())
                    )                      
            else:
                optimizer =  optim_spec(self.masking_network.decoder.parameters())
        else:
            optimizer =  optim_spec(self.masking_network.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer]#,[scheduler]


class resnet50_trained_extractor(masking_network):
    def __init__(self, extractor,  epochs, batch_size, lr =1. , center = False,
                 partition = 1,  noise_mask = False, constrastive = False, inverse = False):
        super().__init__(epochs,lr, batch_size,partition=partition, constrastive=constrastive, noise_mask=noise_mask,supervised=True, inverse=inverse) 
        backbone = extractor
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1, backbone=backbone, 
            freeze_backbone=True, center=center,encoder_channels=encoder_channels,
            decoder_channels=decoder_channels, use_img_space=backbone.use_img_space)
        self.frozen_network = extractor

class medclip_masking_net(masking_network):
    def __init__(self, epochs, batch_size, lr =1. , center = False,
                 partition = 1,  noise_mask = False, constrastive = False,inverse = False, blur = False):
        super().__init__(epochs=epochs,lr=lr, batch_size=batch_size,partition=partition, constrastive=constrastive, noise_mask=noise_mask,supervised=False, inverse=inverse, blur=blur) 
        backbone = medclip_extractor()
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1, backbone=backbone, 
            freeze_backbone=True, center=center,encoder_channels=encoder_channels,
            decoder_channels=decoder_channels, use_img_space=backbone.use_img_space)
        self.frozen_network = backbone
        self.data_mean =[0.485, 0.456, 0.406]
        self.data_std  =[0.229, 0.224, 0.225]