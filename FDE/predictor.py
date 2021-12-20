import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import cv2


class Predictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.__device = device
        checkpoint = torch.load(model_path)
        self.__model = checkpoint['model']
        self.__model.eval()
        self.__model.to(self.__device)

    def run(self, img_path, plot=False):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_np = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR)).astype(np.double)
        rgb_np /= 255
        x = np.zeros([1, 3, 224, 224], dtype=np.float32)
        x[0, :, :, :] = np.transpose(rgb_np, (2, 0, 1))
        x = torch.from_numpy(x).to(self.__device)
        result = self.__model(x)
        output_img = result.detach().cpu().numpy()
        output_img = np.squeeze(output_img)
        if plot:
            plt.imshow(output_img)
            plt.show()
        return output_img


if __name__ == '__main__':
    predictor = Predictor(model_path='../data_in/mobilenet-nnconv5dw-skipadd-pruned.pth.tar')
    predictor.run(img_path='../data_in/rgb.png', plot=True)
