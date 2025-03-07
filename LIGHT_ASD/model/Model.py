import torch
import torch.nn as nn

from model.Classifier import BGRU
from model.Encoder import visual_encoder, audio_encoder
# from Classifier import BGRU
# from Encoder import visual_encoder, audio_encoder

class ASD_Model(nn.Module):
    def __init__(self):
        super(ASD_Model, self).__init__()
        
        self.visualEncoder  = visual_encoder()
        self.audioEncoder  = audio_encoder()
        self.GRU = BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):  
        x = x1 + x2 
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, 128))
        return x    

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, audioFeature, visualFeature):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)  
        outsV = self.forward_visual_backend(visualEmbed)

        return outsAV, outsV
    
    
if __name__ == "__main__":
    model = ASD_Model()
    print(model.named_modules)
    
    inp = torch.randn((1,1,30,224, 224))
    
    out = model(inp)
    print(out)
    print(out.shape)