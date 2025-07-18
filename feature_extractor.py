import torch
import cv2
import torchvision.transforms as T
from torchvision.models import resnet50

class FeatureExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = resnet50(pretrained = True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])


        def extract(self,image):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(tensor).squeeze()
            
            return features.cpu()/features.norm()