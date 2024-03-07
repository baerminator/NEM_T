import torch          
import timm         
from medclip import  MedCLIPVisionModel


    
class resnet50_img_extractor(torch.nn.Module):
    def __init__(
            self,
            model = None,
            
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("resnet50",pretrained=True)
        self.core_model = model
        self.shapes = (224,112 , 56, 28,14, 7,)
        self.channels = (3,64,256, 512, 1024, 2048)
        self.output_shape = (-1,2048,7,7)
        self.use_img_space = True
        
    def forward(self, x: torch.Tensor):
        embs = [x]
        x = self.core_model.conv1(x)
        x = self.core_model.bn1(x)
        x = self.core_model.act1(x)
        embs += [x]
        x = self.core_model.maxpool(x)
       
        x = self.core_model.layer1(x)
        embs += [x]
        x = self.core_model.layer2(x)
        embs += [x]
        x = self.core_model.layer3(x)
        embs += [x]
        x = self.core_model.layer4(x)
        embs += [x]
        return embs
    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)
    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)
    def get_output(self, x: torch.Tensor):
        return self.core_model(x)
        

    
class medclip_extractor(torch.nn.Module):
    def __init__(
            self,
            model = None,
            
    ):
        super().__init__()
        model = MedCLIPVisionModel()
        model.load_from_medclip("/home/gain/work/NEM_MICCAI/pretrained/medclip-resnet")
        self.core_model = model 
        self.shapes = (224,112 , 56, 28,14, 7,)
        self.channels = (3,64,256, 512, 1024, 2048)
        self.output_shape = (-1,2048,7,7)
        self.use_img_space = True
        
    def forward(self, x: torch.Tensor):
        embs = [x]
        x = self.core_model.model.conv1(x)
        x = self.core_model.model.bn1(x)
        x = self.core_model.model.relu(x)
        embs += [x]
        x = self.core_model.model.maxpool(x)
       
        x = self.core_model.model.layer1(x)
        embs += [x]
        x = self.core_model.model.layer2(x)
        embs += [x]
        x = self.core_model.model.layer3(x)
        embs += [x]
        x = self.core_model.model.layer4(x)
        embs += [x]
        return embs
    def get_embeddings(self, x: torch.Tensor):
        return self.core_model(x)