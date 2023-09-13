import datetime
from flask import Flask, Response, jsonify , request
from flask_cors import CORS
import json
import torch
import random
import time
import base64
app = Flask(__name__)
CORS(app)
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import io
from model import *

sleeptime = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.route('/get_data')
def get_loss():

    pretrained = request.args.get('pretrained')
    partially_trained = request.args.get('partially_trained')
    print(pretrained , partially_trained)
    print(type(pretrained) , type(partially_trained))
    if pretrained == "1":
        print("Loading pretrained model")
        saved_model_state_dict = torch.load('backend//G.pth', map_location=torch.device('cpu'))
        print(generator.load_state_dict(saved_model_state_dict))
        saved_model_state_dict = torch.load('backend//D.pth', map_location=torch.device('cpu'))
        print(discriminator.load_state_dict(saved_model_state_dict))
    elif partially_trained == "1":
        print("Loading partially trained model")
        saved_model_state_dict = torch.load('backend//Generator_partial.pth', map_location=torch.device('cpu'))
        print(generator.load_state_dict(saved_model_state_dict))
        saved_model_state_dict = torch.load('backend//Discriminator_partial.pth', map_location=torch.device('cpu'))
        print(discriminator.load_state_dict(saved_model_state_dict))

    def fit(epochs, lr, start_idx=1):
    
        torch.cuda.empty_cache()

        # Create optimizers
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        try:
            for epoch in range(epochs):

                for real_images, _ in tqdm(train_dl):
                    # Train discriminator
                    loss_d, real_score, fake_score , real_images_des , fake_images_des , noise= train_discriminator(real_images, opt_d)

                    data = {
                        "datafrom": "d",
                        "epochs": [],
                        "generatorLoss": [],
                        "discriminatorLoss": [loss_d],
                        "real_score": [real_score],
                        "fake_score": [fake_score],
                        "real_images_des": show_images(real_images_des),
                        "fake_images_des": show_images(fake_images_des),
                        "fake_images_gen": "",
                        "latent": visualize_latent_vector(noise),

                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    # time.sleep(sleeptime)

                    # Train generator
                    loss_g , fake_images_gen , noise = train_generator(opt_g)

                    data = {
                        "datafrom": "g",
                        "epochs": [],
                        "generatorLoss": [loss_g],
                        "discriminatorLoss": [],
                        "real_score": [],
                        "fake_score": [],
                        "real_images_des": "",
                        "fake_images_des": "",
                        "fake_images_gen": show_images(fake_images_gen),
                        "latent": visualize_latent_vector(noise),

                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    # time.sleep(sleeptime)

                    # Loss Data for graph
                    data = {
                        "datafrom": "graph",
                        "epochs": [epoch],
                        "generatorLoss": [loss_g],
                        "discriminatorLoss": [loss_d],
                        "real_score": [],
                        "fake_score": [],
                        "real_images_des": "",
                        "fake_images_des": "",
                        "fake_images_gen": "",
                        "latent": "",

                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    # time.sleep(sleeptime)

        except GeneratorExit:
            print("Generator Exit")
        except Exception as e:
            print(e)

    return Response(fit(epochs , lr ), mimetype='text/event-stream')

if __name__ == '__main__':
    freeze_support()
    device = get_default_device()
    print(device)
    DATA_DIR = 'C://Users//mann1//Desktop//GANS//project//Project//dataset//'
    train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)]))

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    discriminator = to_device(discriminator, device)
    generator = to_device(generator, device)
    app.run(debug=True)
