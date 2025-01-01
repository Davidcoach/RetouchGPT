import os
import re
import time
import torch
import json
import glob
import argparse
import importlib
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP  # 用于分布式
from tqdm import tqdm
from core.lr_scheduler import CosineAnnealingRestartLR
from core.loss_new import AdversarialLoss, VGGLoss, IDLoss, SSIM, AttentionLoss, AlphaClipLoss
from core.dataset import FaceDataset
import lpips, wandb
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

parser = argparse.ArgumentParser(description='RetouchGPT')
parser.add_argument('-c', '--config', default='RetouchGPT/config/retouchgpt.json', type=str)
args = parser.parse_args()

config = json.load(open(args.config))

def get_ip():
    # 从环境变量获取主节点名称
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    if not node_list:
        print("Qrong!")
        return None

    # 使用正则表达式匹配第一个出现的两位数字
    match = re.search(r"\d{2}", node_list)
    if match:
        first_two_digits = match.group(0)  # 获取匹配到的第一组两位数字
        return 'gpu' + first_two_digits
    else:
        print("Wrong！")
        return None


def load_model(rank, netG, netD, scheG, scheD, optimG, optimD):
    """Load netG (and netD)."""
    # get the latest checkpoint
    model_path = config['save_dir']
    print("3. pretrained model are loaded")
    if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
        latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
    else:
        ckpts = [
            os.path.basename(i).split('.pth')[0]
            for i in glob.glob(os.path.join(model_path, '*.pth'))
        ]
        ckpts.sort()
        latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

    if latest_epoch is not None:
        gen_path = os.path.join(model_path,
                                f'gen_{int(latest_epoch):06d}.pth')
        dis_path = os.path.join(model_path,
                                f'dis_{int(latest_epoch):06d}.pth')
        opt_path = os.path.join(model_path,
                                f'opt_{int(latest_epoch):06d}.pth')

        if rank == 0:
            print(f'Loading model from {gen_path}...')
        dataG = torch.load(gen_path, map_location="cpu")
        netG.load_state_dict(dataG, strict=True)
        del dataG
        if not config['model']['no_dis']:
            dataD = torch.load(dis_path,
                               map_location="cpu")
            netD.load_state_dict(dataD)
            del dataD
        data_opt = torch.load(opt_path, map_location="cpu")
        optimG.load_state_dict(data_opt['optimG'])
        scheG.load_state_dict(data_opt['scheG'])
        optimD.load_state_dict(data_opt['optimD'])
        scheD.load_state_dict(data_opt['scheD'])
        epoch = data_opt['epoch']
        iteration = data_opt['iteration']
    else:
        epoch = 0
        iteration = 0
    return netG, netD, scheG, scheD, optimG, optimD, epoch, iteration


def pair(source_tensor, target_tensor, abnormal_txt, netG, netD, optimD, optimG, local_rank, world_size, loss_functions):
    b, c, h, w = source_tensor.size()
    pred_imgs, output_abnormal, mask, gen_acc = netG(source_tensor, abnormal_txt, local_rank)
    pred_imgs = pred_imgs.view(b, c, h, w)

    gen_loss = 0
    dis_loss = 0
    # loss

    # 使用字典中的损失函数
    if not config['model']['no_dis']:
        real_clip = netD(target_tensor)
        fake_clip = netD(pred_imgs.detach())

        # 使用字典中的 adversarial_loss
        dis_real_loss = loss_functions["adversarial_loss"](real_clip, True, True)
        dis_fake_loss = loss_functions["adversarial_loss"](fake_clip, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        optimD.zero_grad()
        dis_loss.backward()
        optimD.step()

        # generator adversarial loss
        gen_clip = netD(pred_imgs)
        gan_loss = loss_functions["adversarial_loss"](gen_clip, True, False)
        gan_loss = gan_loss * config['losses']['adversarial_weight']
        gen_loss += gan_loss

    # generator l1 loss
    valid_loss = loss_functions["l1_loss_func"](pred_imgs, target_tensor)
    valid_loss = valid_loss * config['losses']['valid_weight']
    gen_loss += valid_loss

    # mask attention loss
    mask_atten_loss = loss_functions["attention_loss_func"](mask, source_tensor, target_tensor)
    mask_atten_loss = mask_atten_loss * config['losses']['mask_weight']
    gen_loss += mask_atten_loss

    # lpips loss
    lpips_loss = loss_functions["lpips_loss_func"].forward(pred_imgs, target_tensor).mean()
    lpips_loss = lpips_loss * config['losses']['lpips_weight']
    gen_loss += lpips_loss

    # ssim loss
    ssim_loss = (1 - loss_functions["ssim_loss_func"](pred_imgs, target_tensor))
    ssim_loss = ssim_loss * config['losses']['ssim_weight']
    gen_loss += ssim_loss

    # clip loss
    clip_loss = loss_functions["clip_loss_func"](pred_imgs, mask)
    clip_loss = clip_loss * config['losses']['clip_weight']
    gen_loss += clip_loss

    # 将所有进程的 running_loss 求和
    running_loss_tensor = torch.tensor([gen_loss], device='cuda:%d' % local_rank)
    torch.distributed.all_reduce(running_loss_tensor, op=torch.distributed.ReduceOp.SUM)
    # 计算平均 loss
    average_loss = running_loss_tensor.item() / world_size

    optimG.zero_grad()
    gen_loss.backward()
    optimG.step()
    return dis_loss, valid_loss, lpips_loss, ssim_loss, gen_loss, gen_acc, mask_atten_loss, clip_loss


def get_lr(scheG):
    return scheG.get_lr()[0]

def save(rank, iteration, epoch, netG, netD, optimD, optimG, scheG, scheD):
    if rank==0:
        gen_path = os.path.join(config['save_dir'], f'gen_{iteration:06d}.pth')
        dis_path = os.path.join(config['save_dir'], f'dis_{iteration:06d}.pth')
        opt_path = os.path.join(config['save_dir'], f'opt_{iteration:06d}.pth')
        print(f'\nsaving model to {gen_path} ...')

        # remove .module for saving
        if isinstance(netG, torch.nn.DataParallel) or isinstance(netG, DDP):
            netG = netG.module
            if not config['model']['no_dis']:
                netD = netD.module
        else:
            netG = netG
            if not config['model']['no_dis']:
                netD = netD

        # save checkpoints
        torch.save(netG.state_dict(), gen_path)
        if not config['model']['no_dis']:
            torch.save(netD.state_dict(), dis_path)
            torch.save(
                {
                    'epoch': epoch,
                    'iteration': iteration,
                    'optimG': optimG.state_dict(),
                    'optimD': optimD.state_dict(),
                    'scheG': scheG.state_dict(),
                    'scheD': scheD.state_dict(),
                    # 'scaler': scaler.state_dict()
                }, opt_path)
        else:
            torch.save(
                {
                    'epoch': epoch,
                    'iteration': iteration,
                    'optimG': optimG.state_dict(),
                    'scheG': scheG.state_dict()
                }, opt_path)
        latest_path = os.path.join(config['save_dir'], 'latest.ckpt')
        os.system(f"echo {iteration:06d} > {latest_path}")


def test(iteration, netG, test_loader, local_rank, lr):
    netG.eval()
    cnt = 0
    PSNR = 0
    SSIM = 0
    LPIPS = 0
    ACC = 0
    device = config['device']
    loss_fn = lpips.LPIPS(net='alex').to(device)
    for source_tensor, target_tensor, abnormal_txt, normal_txt in tqdm(test_loader):
        with torch.no_grad():
            source_tensor = source_tensor.cuda(local_rank)
            target_tensor = target_tensor.cuda(local_rank)
            abnormal_txt = abnormal_txt.cuda(local_rank)
            result, output_abnormal, mask, gen_acc = netG(source_tensor, abnormal_txt)
            lpips_loss = loss_fn(result, target_tensor).mean()
            s_img = result[0].cpu().numpy()
            t_img = target_tensor[0].cpu().numpy()
            psnr = compare_psnr(t_img, s_img)
            ssim = compare_ssim(t_img, s_img, channel_axis=0, data_range=2)
            PSNR += psnr
            SSIM += ssim
            LPIPS += lpips_loss
            ACC += gen_acc
            cnt += 1
    PSNR /= cnt
    SSIM /= cnt
    LPIPS /= cnt
    ACC /= cnt
    print(iteration, "PSNR: ", PSNR, "SSIM:", SSIM, "ACC: ", ACC, "LPIPS: ", LPIPS)
    if config["trainer"]["use_wandb"] == 1:
        wandb.log({"SSIM": SSIM, "ACC": ACC, "PSNR": PSNR, "LPIPS": LPIPS})
    eval_txt = config['eval_txt']
    with open(eval_txt, 'a') as f:
        f.writelines(f"lr: {lr}; {iteration}: PSNR: {PSNR}, SSIM: {SSIM};  ACC: {ACC}; LPIPS: {LPIPS}\n")
    f.close()
    netG.train()

def _train_epoch(pbar, train_loader, iteration, netG, netD, optimD, optimG, scheG, scheD, local_rank, rank, world_size, test_loader, loss_functions):
    """Process input and calculate loss every training epoch"""
    print("4. start training")
    for source_tensor, target_tensor, abnormal_txt, normal_txt in train_loader:
        source_tensor = source_tensor.cuda(local_rank)
        target_tensor = target_tensor.cuda(local_rank)
        iteration += 1
        dis_loss, valid_loss, lpips_loss, ssim_loss, gen_loss, gen_acc, mask_atten_loss, clip_loss = pair(source_tensor, target_tensor, abnormal_txt, netG, netD, optimD, optimG, local_rank, world_size, loss_functions)
        if iteration % 20e3 == 0:
            scheG.step()
            scheD.step()
        if rank == 0:
            pbar.update(1)
            lr = get_lr(scheG)
            pbar.set_description((f"d: {dis_loss.item():.3f}; "
                                  f"valid: {valid_loss.item():.3f}; "
                                  f"clip_loss: {clip_loss.item():.3f}; "
                                  f"lr: {lr:.6f}"
                                  ))

            if config["trainer"]["use_wandb"] == 1:
                wandb.log({
                    "dis": dis_loss.item(),
                    "valid": valid_loss.item(),
                    "lpips_loss": lpips_loss.item(),
                    "mask_atten_loss": mask_atten_loss.item(),
                    "ssim": ssim_loss.item(),
                    "clip_loss": clip_loss.item(),
                    "gen_acc": gen_acc,
                    "lr": lr
                })
        # saving models
        if iteration % config['trainer']['save_freq'] == 0:
            save(int(iteration))
            if iteration == 2e4:
                test(iteration, netG, test_loader, local_rank, lr=get_lr(scheG))
            if iteration == 6e4:
                test(iteration, netG, test_loader, local_rank, lr=get_lr(scheG))
            if iteration == 10e4:
                test(iteration, netG, test_loader, local_rank, lr=get_lr(scheG))
            if iteration == 13e4:
                test(iteration, netG, test_loader, local_rank, lr=get_lr(scheG))
            if iteration > 15e4 - 1:
                test(iteration, netG, test_loader, local_rank, lr=get_lr(scheG))
        if iteration > config['trainer']['iterations']:
            break


def main(config):
	# 初始化分布式环境
    os.environ['MASTER_ADDR'] = get_ip()
    os.environ['MASTER_PORT'] = '14231'
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank= int(os.environ['SLURM_LOCALID'])
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(world_size, local_rank, rank)

    train_set = FaceDataset(path=config['train_data_loader']['dataroot'], resolution=512, data_type="train", return_mask=False)
    sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['trainer']['batch_size'], sampler=sampler)

    val_set = FaceDataset(path=config['train_data_loader']['dataroot'], resolution=512, data_type="test", return_mask=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # models
    model = importlib.import_module('model.' + config['model']['net'])
    netG = model.InpaintGenerator().cuda(local_rank)
    # netG.conv.requires_grad_(False)
    netG = DDP(netG, device_ids=[local_rank])# , find_unused_parameters=True)
    netD = model.Discriminator().cuda(local_rank)
    netD = DDP(netD, device_ids=[local_rank])# , find_unused_parameters=True)
    if config["trainer"]["use_wandb"] == 1:
        wandb.init(project="retouching", name=config['model']['net'] + "_DDP")
    print("1. models are initialized.")

    # optimizers
    backbone_params = []
    for name, param in netG.named_parameters():
        if param.requires_grad == True:
            backbone_params.append(param)
    optim_params = [
        {
            'params': backbone_params,
            'lr': config['trainer']['lr']
        }
    ]
    optimG = torch.optim.Adam(optim_params, betas=(config['trainer']['beta1'], config['trainer']['beta2']))
    optimD = torch.optim.Adam(netD.parameters(), betas=(config['trainer']['beta1'],config['trainer']['beta2']))

    # schedulers
    scheduler_opt = config['trainer']['scheduler']
    scheG = CosineAnnealingRestartLR(optimG, periods=scheduler_opt['periods'], restart_weights=scheduler_opt['restart_weights'])
    scheD = CosineAnnealingRestartLR(optimD, periods=scheduler_opt['periods'], restart_weights=scheduler_opt['restart_weights'])
    print("2. optimizer and scheduler are initialized")

    netG, netD, scheG, scheD, optimG, optimD, epoch, iteration = load_model(rank, netG, netD, scheG, scheD, optimG, optimD)

    pbar = range(int(config['trainer']['iterations']))
    if rank == 0:
        pbar = tqdm(pbar, initial=iteration, dynamic_ncols=True, smoothing=0.01)

    optimG.zero_grad()
    optimG.step()
    optimD.zero_grad()
    optimD.step()

    loss_functions = {
        "adversarial_loss": AdversarialLoss(type=config['losses']['GAN_LOSS']).cuda(local_rank),
        "l1_loss_func": nn.SmoothL1Loss().cuda(local_rank),
        "lpips_loss_func": lpips.LPIPS(net='alex', lpips=False).cuda(local_rank),
        "vgg_loss_func": VGGLoss(local_rank),
        "attention_loss_func": AttentionLoss(),
        "ID_loss_func": IDLoss(local_rank),
        "ssim_loss_func": SSIM(window_size=11),
        "clip_loss_func": AlphaClipLoss(local_rank).cuda(local_rank)}


    while True:
        epoch += 1
        sampler.set_epoch(epoch)
        _train_epoch(pbar, train_loader, iteration, netG, netD, optimD, optimG, scheG, scheD, local_rank, rank, world_size, val_loader, loss_functions)
        if iteration > config['trainer']['iterations']:
            save(int(iteration))
            break
    print('\nEnd training....')


if __name__ == '__main__':
    main(config)