from PIL import Image
import torch
from torch import nn
from torchvision import models,transforms
trained_model = None
class_names = ['FAKE','REAL']

class deepfakeclassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(64*28*28, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
    def forward(self,x):
        x = self.network(x)
        return(x)
    

def predict(image_path):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    global trained_model 

    if trained_model is None:
        trained_model = deepfakeclassifier()
        trained_model.load_state_dict(torch.load('saved_model2-98.70.pth'))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _,predicted_class = torch.max(output,1)
        return class_names[predicted_class.item()]
