## Gradient Based Saliency
## Demo By: Rahul Andurkar
## 5/11/2022

## Quick gradient based saliency test based on K. Simonyan 2014
## and articles pertaining to it

import torch
from torchvision.models import resnet50
from torchvision.models import efficientnet_b7
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import urllib.request

class GBSVisualizer():
    ## Constructor defining the model and transformations
    def __init__(self):
        self.model = efficientnet_b7(pretrained=True)
        self.fix_model()
        self.model.eval()

        self.T_input = transforms.Compose([
            transforms.Resize((600,600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.T_output = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ])
        ])

    ## Grab gradient at every relu layer
    def relu_hook_function(self, module, grad_in, grad_out):
        if isinstance(module, torch.nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.),)

    ## Call hooks on backpropagation
    def fix_model(self):
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(self.relu_hook_function)

    ## Load and preprocess image
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        ## Resize image tensor
        preprocessed_image = self.T_input(image)
        preprocessed_image =  preprocessed_image.reshape(1, 3, 600, 600)

        return preprocessed_image

    def decode_output(self, output):
        # Read the categories
        url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        urllib.request.urlretrieve(url, "imagenet1000_clsidx_to_labels.txt")
        # Read the categories
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_objid = torch.topk(probabilities, 413)
        #for i in range(top5_prob.size(0)):
        #    print(categories[top5_objid[i]], top5_prob[i].item())

        return top5_objid[412]

    def preprocess_graident(self, gradient):
        preprocessed_grad = self.T_output(gradient)[0]
        preprocessed_grad = preprocessed_grad.detach().numpy().transpose(1, 2, 0)

        return preprocessed_grad

    def make_map(self, gradient):
        map = self.preprocess_graident(gradient)
        map = map[:, :, 0] + map[:, :, 1] + map[:, :, 2]
        map = (map - np.min(map)) / (np.max(map) - np.min(map))

        return map

    ## Execute whole process
    def run(self, image_path):
        ## Preprocess image
        image = self.preprocess_image(image_path)
        image.requires_grad = True

        ## Forward propagation
        output = self.model(image)

        ## Backpropagation
        best_id = self.decode_output(output)
        output[0, best_id].backward()
        gradients = image.grad

        gbsmap = self.make_map(gradients)

        img = cv2.imread(img_path)

        gbsmap = cv2.resize(gbsmap, (img.shape[1], img.shape[0]))
        gbsmap = np.uint8(255 * gbsmap)
        gbsmap = cv2.normalize(gbsmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gbsmap = cv2.applyColorMap(gbsmap, cv2.COLORMAP_JET)
        ret, gbsmap = cv2.threshold(gbsmap, 250, 255, cv2.THRESH_BINARY)

        rows,cols,_ = gbsmap.shape

        for i in range(rows):
            for j in range(cols):
                if np.array_equal(gbsmap[i][j], [0, 255, 0]):
                    gbsmap[i][j] = img[i][j]

        superimposed_img = cv2.addWeighted(gbsmap, 1, img, 0.25, 0)
        cv2.imwrite(r'C:\Users\rahul\Downloads\GradientBasedSaliency\GBS_Images\superimposed.png', superimposed_img)
        cv2.imwrite(r'C:\Users\rahul\Downloads\GradientBasedSaliency\GBS_Images\rawmap.png', gbsmap)

    def testgradcam(self, image_path):
        ## Preprocess image
        image = self.preprocess_image(image_path)
        image.requires_grad = True

        ## Forward propagation
        output = self.model(image)

        # get the gradient of the output with respect to the parameters of the model
        output[:, 413].backward()

        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.model.get_activations(image).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())

if __name__ == '__main__':
    img_path = r'C:\Users\rahul\Downloads\GradientBasedSaliency\gaze_gesture_ego_view.png'
    gbs = GBSVisualizer()
    #gbs.run(img_path)
    gbs.testgradcam(img_path)