import torch
from options import TrainOptions
import numpy as np
from PIL import Image
from torchvision import transforms
import process
import RPTC
import os

model_path = "RPTC.pth"

opt = TrainOptions().parse(print_option=False)

model = RPTC.Net()
model.apply(RPTC.initWeight)

state_dict = torch.load(model_path, map_location='cpu')

model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})
# model.load_state_dict(state_dict['model'])
# model.cuda()
model.eval()

img_path = "D:/picture_for_test/fake/fake3.jpg"

img = Image.open(img_path).convert("RGB")
input_tensor = process.processing_RPTC(img, opt).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    output = model(input_tensor)

prediction = torch.sigmoid(output).item()

print(prediction)
