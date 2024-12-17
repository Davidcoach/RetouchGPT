# -*- coding: utf-8 -*-
import importlib
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from core.dataset import Testset
from torchvision.utils import save_image
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="RetouchGPT")
parser.add_argument("-e", "--epoch", type=str, default="320000")
parser.add_argument("-c", "--ckpt", type=str, default= "./release_model/RetouchGPT_retouchgpt")
parser.add_argument("--size",  type=int, default=512)
parser.add_argument("--model", type=str, default='RetouchGPT')
parser.add_argument("--input_path", type=str, required=True, action="The input path of in-the-wild images")
parser.add_argument("--save_path", type=str, default= "results/test_atten")
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # First load the chechpoints of RetouchGPT
    module_name = 'model.' + args.model
    net = importlib.import_module(module_name)
    model = net.InpaintGenerator().to(device)
    print(args.model)
    data = torch.load("{0}/gen_{1}.pth".format(args.ckpt, args.epoch), map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format("{0}/gen_{1}.pth".format(args.ckpt, args.epoch)))
    model.eval()

    # Load the testset of RetouchGPT
    test_dataset = Testset(args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Set up the save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    for name, source_tensor in tqdm(test_loader):
        with torch.no_grad():
            # user instructions
            prompt_text = "Human: Is there any imperfection in the image? Please retouch the image."
            save_path = os.path.join(save_path, f"{name[0]}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Forward RetouchGPT
            pred_img, output_text, masks = model(source_tensor.to(device), prompt_text)

            # Save output results
            path = os.path.join(save_path, f"{str(name[0])}_out.png")
            save_image(pred_img, path, normalize=True, value_range=(-1, 1))
            with open(f"output_response.txt", "a") as f:
                f.writelines(f"{name[0]}, \n {prompt_text}, \n, gpt: {output_text}, \n ")
            mask_path = os.path.join(save_path, f"{str(name[0])}_mask.png")
            save_image(masks[0], mask_path, normalize=True, value_range=(0, 1))