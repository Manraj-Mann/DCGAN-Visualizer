import torch
from model import *

saved_model_state_dict = torch.load('backend//G.pth', map_location=torch.device('cpu'))
print(generator.load_state_dict(saved_model_state_dict))
saved_model_state_dict = torch.load('backend//D.pth', map_location=torch.device('cpu'))
print(discriminator.load_state_dict(saved_model_state_dict))
