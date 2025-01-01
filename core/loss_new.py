import torchvision
from core.model_irse import Backbone
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
from math import exp
import clip
from alpha_clip import alpha_clip
import numpy as np


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


class VGGLoss(nn.Module):
    def __init__(self, local_rank, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        vgg = torchvision.models.vgg19(pretrained=True).features
        # vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.cuda(local_rank))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().cuda(local_rank)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.atten_loss = torch.nn.L1Loss()

    def sigmoid(self, x, fac=50):
        return 1 / (1 + torch.exp(-fac * x))

    def forward(self, attentions, input_img, gt_img, mask=None):
        diff = torch.abs(input_img - gt_img).mean(dim=1, keepdim=True)
        if mask is not None:
            diff = diff * mask
        diff_normed = (diff - diff.amin(dim=[2, 3], keepdim=True)) / (diff.amax(dim=[2, 3], keepdim=True) - diff.amin(dim=[2, 3], keepdim=True))
        diff_normed = self.sigmoid(diff_normed)
        # diff_normed[diff_normed==0.5] = 0.1
        # utils.save_image(input_img, f"source.png", normalize=True, range=(-1, 1))
        # utils.save_image(gt_img, f"target.png", normalize=True, range=(-1, 1))
        loss = 0
        for atten in attentions:
            res = atten.shape[2:]
            diff_normed_res = F.interpolate(diff_normed, res, mode='bilinear', align_corners=True)
            loss += self.atten_loss(atten, diff_normed_res)
        # exit()
        return loss


class IDLoss(nn.Module):
    def __init__(self, local_rank, ckpt_dict=None):
        super(IDLoss, self).__init__()
        # print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').cuda(local_rank)
        if ckpt_dict is None:
            self.facenet.load_state_dict(
                torch.load("model/modules/model_ir_se50.pth", map_location=torch.device('cuda')))
        else:
            self.facenet.load_state_dict(ckpt_dict)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        # print("done")

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h == w
        ss = h // 256
        x = x[:, :, 35 * ss:-33 * ss, 32 * ss:-36 * ss]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        # sim_improvement = 0
        # id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            # diff_input = y_hat_feats[i].dot(x_feats[i])
            # diff_views = y_feats[i].dot(x_feats[i])
            # id_logs.append({'diff_target': float(diff_target),
            #                 'diff_input': float(diff_input),
            #                 'diff_views': float(diff_views)})
            loss += 1 - diff_target
            # id_diff = float(diff_target) - float(diff_views)
            # sim_improvement += id_diff
            count += 1

        return loss / count


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / torch.sum(gauss)  # 归一化


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # window_size,1
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def sigmoid(x, fac=60):
    return 1 / (1 + torch.exp(-fac * (x)))


def cal_entropy(x):
    idx1 = x <= 0
    idx2 = x > 1
    idx3 = torch.logical_and(x > 0, x <= 1)

    x1 = x * 0.0
    x2 = x * 0.0
    x3 = -x * torch.log(x + 1e-12)

    x = x1 * idx1 + x2 * idx2 + x3 * idx3
    return x


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).type_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


def calc_contrastive_loss(query, key, queue, temp=0.07):
    N = query.shape[0]
    K = queue.shape[0]

    zeros = torch.zeros(N, dtype=torch.long, device=query.device)
    key = key.detach()
    logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
    logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

    logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

    loss = F.cross_entropy(logit / temp, zeros)

    return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        # self.vgg = Vgg19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # Compute loss
        weight = [1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
        style_loss = 0.
        for i in range(len(x)):
            style_loss += self.criterion(self.compute_gram(x[i]), self.compute_gram(y[i])) * weight[i]
        return style_loss


class ClipLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(ClipLoss, self).__init__()
        self.device = device
        self.loss = nn.L1Loss()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # self.text = ["A person with 'dark circle'",
        # "A person with 'forehead wrinkles'",
        # "A person with 'Acne'",
        # "A person with 'freckle'",
        # "A person with 'pimple'",
        # "A person with 'blemish'",
        # "A person with 'imperfection'",
        # "A person with 'flaw'",
        # "A person with 'yellow teeth'",
        # "A person with 'black teeth'",
        # "A person with 'crow's feet'",
        # "A person with 'fine lines'",
        # "A person with 'laugh line'",
        # "A person with 'nasolabial folds'",
        # "A person with 'smile line'",
        # "A person with 'shadow'",
        # "A person with 'eye bag'",
        # "A person with 'blackhead'",
        # "A person with 'cervical stripe'",
        # "A person with 'liver spot'"]
        self.text = [
            "A person with 'dark circle' under the eyes. Dark circles are often described as a discoloration or pigmentation of the under-eye area. They can appear as a dark, blue-gray or purple hue and are usually more prominent in people with lighter skin tones.",
            "A person with 'wrinkle' on the forehead. Wrinkles are lines or creases that form on the skin as a natural part of the aging process. They can appear as fine lines, deep creases, or folds on forehead.",
            "A person with 'pimple' on the right cheek. A pimple appears as a raised red bump on right cheek, often with a white or yellowish center.",
            "A person with 'Acne'. Acne refers to a common skin condition characterized by the presence of pimples, blackheads, whiteheads, and sometimes cysts or nodules on the skin.",
            "A person with 'freckle'. Freckles are small, flat spots on the skin that are typically light brown or tan in color. They are usually round, but can vary in shape and size.",
            "A person with 'blemish'. Blemish is a general term for any type of mark, spot, discoloration, or imperfection on the skin. It can refer to a wide range of skin conditions, including acne, pimples, blackheads, whiteheads, age spots, sun spots, and scars.",
            "A person with 'imperfection'. Imperfections can refer to various flaws or abnormalities that can be found in different contexts, such as in art, objects, or even in human appearance.",
            "A person with 'flaw'. Flaws typically refer to imperfections or defects that reduce the quality or value of something. Similar to imperfections, the appearance of a flaw can vary depending on the context in which it is being considered.",
            "A person with 'yellow teeth'. Yellow teeth refers to teeth that have a noticeable yellowish discoloration. The color of teeth can vary naturally from person to person, but generally, healthy teeth are considered to be shades of white.",
            "A person with 'black teeth'. Black teeth refers to teeth that have a dark or blackened appearance. It is a condition that is typically caused by severe tooth decay or trauma to the tooth.",
            "A person with 'crows feet'. Crows feet is a term used to describe the fine lines or wrinkles that appear at the outer corners of the eyes. They are called crows feet because the lines resemble the feet of a crow.",
            "A person with 'fine lines'. Fine lines is a term used to describe small, shallow wrinkles on the skins surface. These lines are usually thinner and less noticeable compared to deeper wrinkles.",
            "A person with 'laugh line'. Laugh lines also known as smile lines or nasolabial folds, are facial lines that appear around the mouth and nose. These lines are typically the result of frequent smiling, laughing, and other facial expressions over time.",
            "A person with 'nasolabial folds'. Nasolabial folds also known as smile lines or laugh lines, are facial lines that extend from the sides of the nose to the corners of the mouth.",
            "A person with 'smile line'. The term smile line is often used interchangeably with nasolabial folds to describe the facial lines that form from the sides of the nose to the corners of the mouth.",
            "A person with 'shadow'. A human shadow refers to the silhouette or outline formed by an individual when they are illuminated by a light source. ",
            "A person with 'eye bag'. Eye bags are a common term used to describe the swelling or puffiness that occurs directly under the eyes. They can vary in appearance and severity.",
            "A person with 'blackhead'. A blackhead is a type of acne lesion that is characterized by a small, raised bump on the skin.",
            "A person with 'cervical stripe'. Cervical stripe refers to a specific feature or condition related to the neck or cervical spine.",
            "A person with 'liver spot'. A liver spot is a common term used to describe a type of hyperpigmentation or dark spots on the skin."]

    def forward(self, imgs):
        loss = 0
        for pred_img in imgs:
            pred_img = transforms.ToPILImage()(pred_img)
            image = self.preprocess(pred_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # image_features = self.model.encode_image(image)
                # text_features = self.model.encode_text(text)
                for t in self.text:
                    t = clip.tokenize(t).to(self.device)
                    logits_per_image, logits_per_text = self.model(image, t)
                    loss += torch.mean(logits_per_image)
            return loss


def cal_softmIou(x, y):
    x_gz = x * 255
    x_gz = (x_gz > 128)
    y_gz = y * 255
    y_gz = (y_gz > 128)
    inter_idx = x_gz & y_gz
    inter_sum = ((x + y) / 2)[inter_idx].sum()
    union_sum = (x[x_gz].sum() + y[y_gz].sum()) - inter_sum
    return inter_sum / union_sum


class AlphaClipLoss(nn.Module):
    def __init__(self, local_rank):
        super(AlphaClipLoss, self).__init__()
        self.model, self.preprocess = alpha_clip.load("ViT-B/16",
                                                      alpha_vision_ckpt_pth="model/modules/clip_b16_grit1m_fultune_8xe.pth",
                                                      device=f"cuda: {local_rank}")  # change to your own ckpt path
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])
        self.model = self.model.float()
        self.text = [
            "A person with 'dark circle' under the eyes. Dark circles are often described as a discoloration or pigmentation of the under-eye area. They can appear as a dark, blue-gray or purple hue and are usually more prominent in people with lighter skin tones.",
            "A person with 'wrinkle' on the forehead. Wrinkles are lines or creases that form on the skin as a natural part of the aging process. They can appear as fine lines, deep creases, or folds on forehead.",
            "A person with 'pimple' on the right cheek. A pimple appears as a raised red bump on right cheek, often with a white or yellowish center.",
            "A person with 'Acne'. Acne refers to a common skin condition characterized by the presence of pimples, blackheads, whiteheads, and sometimes cysts or nodules on the skin.",
            "A person with 'freckle'. Freckles are small, flat spots on the skin that are typically light brown or tan in color. They are usually round, but can vary in shape and size.",
            "A person with 'blemish'. Blemish is a general term for any type of mark, spot, discoloration, or imperfection on the skin. It can refer to a wide range of skin conditions, including acne, pimples, blackheads, whiteheads, age spots, sun spots, and scars.",
            "A person with 'imperfection'. Imperfections can refer to various flaws or abnormalities that can be found in different contexts, such as in art, objects, or even in human appearance.",
            "A person with 'flaw'. Flaws typically refer to imperfections or defects that reduce the quality or value of something. Similar to imperfections, the appearance of a flaw can vary depending on the context in which it is being considered.",
            "A person with 'yellow teeth'. Yellow teeth refers to teeth that have a noticeable yellowish discoloration. The color of teeth can vary naturally from person to person, but generally, healthy teeth are considered to be shades of white.",
            "A person with 'black teeth'. Black teeth refers to teeth that have a dark or blackened appearance. It is a condition that is typically caused by severe tooth decay or trauma to the tooth.",
            "A person with 'crows feet'. Crows feet is a term used to describe the fine lines or wrinkles that appear at the outer corners of the eyes. They are called crows feet because the lines resemble the feet of a crow.",
            "A person with 'fine lines'. Fine lines is a term used to describe small, shallow wrinkles on the skins surface. These lines are usually thinner and less noticeable compared to deeper wrinkles.",
            "A person with 'laugh line'. Laugh lines also known as smile lines or nasolabial folds, are facial lines that appear around the mouth and nose. These lines are typically the result of frequent smiling, laughing, and other facial expressions over time.",
            "A person with 'nasolabial folds'. Nasolabial folds also known as smile lines or laugh lines, are facial lines that extend from the sides of the nose to the corners of the mouth.",
            "A person with 'smile line'. The term smile line is often used interchangeably with nasolabial folds to describe the facial lines that form from the sides of the nose to the corners of the mouth.",
            "A person with 'shadow'. A human shadow refers to the silhouette or outline formed by an individual when they are illuminated by a light source. ",
            "A person with 'eye bag'. Eye bags are a common term used to describe the swelling or puffiness that occurs directly under the eyes. They can vary in appearance and severity.",
            "A person with 'blackhead'. A blackhead is a type of acne lesion that is characterized by a small, raised bump on the skin.",
            "A person with 'cervical stripe'. Cervical stripe refers to a specific feature or condition related to the neck or cervical spine.",
            "A person with 'liver spot'. A liver spot is a common term used to describe a type of hyperpigmentation or dark spots on the skin."]

    def forward(self, pred_img, mask):
        loss = 0
        pred_img = transforms.ToPILImage()(pred_img[0]).convert('RGB')
        mask = np.array(transforms.ToPILImage()(mask[0][0]))
        mask = self.mask_transform((mask * 255).astype(np.uint8))
        mask = mask.cuda().unsqueeze(dim=0)

        image = self.preprocess(pred_img).unsqueeze(0).cuda()
        text = alpha_clip.tokenize(self.text).cuda()
        with torch.no_grad():
            image_features = self.model.visual(image, mask)
            text_features = self.model.encode_text(text)

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        ## print the result
        similarity = (100.0 * image_features @ text_features.T)

        loss = torch.mean(similarity)

        return loss
