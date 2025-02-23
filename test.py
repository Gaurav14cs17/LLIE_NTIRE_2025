import os
import argparse
import torch
import cv2
import numpy as np
from model import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--is_save', action='store_true', default=True,help='Save images')
    parser.add_argument('--data_in_path', type=str, default="./Data/Val_Input_Low_Light_2025/Input/")
    parser.add_argument('--data_output_dir_path', type=str, default='./model_output/')
    parser.add_argument('--ckpt_path', type=str, default='./LYT_Torch_Weights/best_model_LOLv1.pth', help='Path to model checkpoint')
    args = parser.parse_args()
    return args


class InfranceCode:
    def __init__(self):
        self.args = parse_args()
        self.test_file_path = self.args.data_in_path

        # Ensure output directory exists if saving is enabled
        if self.args.is_save:
            self.output_path = self.args.data_output_dir_path
            os.makedirs(self.output_path, exist_ok=True)
            print("File Created !!")

        # Load the model
        self.get_model()

    def get_model(self):
        """Loads the pre-trained model from checkpoint"""
        # Define the model architecture (should match the saved checkpoint model)
        self.model = LYT()
        # Load model weights
        weight = torch.load(self.args.ckpt_path,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(weight)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()

    def infrance_single_img(self, img):
        """Processes a single image using the model"""
        _, _, h_old, w_old = img.shape  # Corrected .size() to .shape

        # Padding to ensure proper dimensions
        padding = 16 * 2
        h_pad = (h_old // padding + 1) * padding - h_old
        w_pad = (w_old // padding + 1) * padding - w_old

        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]

        img = self.tile_eval(img, tile=384, tile_overlap=96)
        img = img[..., :h_old, :w_old]
        return img

    def tile_eval(self, input_, tile=128, tile_overlap=32):
        """Splits the image into overlapping tiles and processes them separately"""
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "Tile size should be a multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                #print("out_patch : ", in_patch.shape)
                out_patch = self.model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)

        restored = E.div_(W)
        restored = torch.clamp(restored, 0, 1)
        return restored

    def post_process(self, model_output):
        """Converts the model output tensor to an image"""
        model_output = model_output.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        model_output = (model_output * 255.).round().clip(0, 255).astype(np.uint8)
        model_output = cv2.cvtColor(model_output, cv2.COLOR_RGB2BGR)
        return model_output

    def pre_process(self, in_image):
        """Prepares the input image for the model"""
        img = np.ascontiguousarray(in_image.transpose((2, 0, 1)))
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)
        return img

    def save_image(self, image_name, output_image):
        """Saves the output image"""
        save_path = os.path.join(self.output_path, image_name.replace('.jpg', '.png'))
        cv2.imwrite(save_path, output_image)

    def run(self):
        """Runs inference on all images in the input directory"""
        with torch.no_grad():
            for image_name in sorted(os.listdir(self.test_file_path)):
                image_path = os.path.join(self.test_file_path, image_name)
                in_image = cv2.imread(image_path)

                if in_image is None:
                    print(f"Skipping invalid image: {image_name}")
                    continue

                in_image = self.pre_process(in_image)
                img = in_image.cuda() if torch.cuda.is_available() else in_image

                model_output = self.infrance_single_img(img)
                model_output = model_output.clamp_(0, 1)
                model_output = self.post_process(model_output)

                if self.args.is_save:
                    print("Save : ", image_name)
                    self.save_image(image_name, model_output)


if __name__ == '__main__':
    model_obj = InfranceCode()
    model_obj.run()
