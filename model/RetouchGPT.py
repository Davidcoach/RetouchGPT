''' RetouchGPT: LLM-based Interactive High-Fidelity Face Retouching via Imperfection Prompting
'''
import torch
import torch.nn as nn
from model.spectral_norm import spectral_norm as _spectral_norm
import torch.nn.functional as F
from torch.nn.utils import rnn
from transformers import LlamaTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from model.network import Stage
from model.enc_dec import Decoder, Encoder, EqualLinear, ConvLayer

PROMPT_START = '### Human: <Img>'


def T5_model_load(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    for param in model.parameters():
        param.requires_grad = False

    print('T5 model initialized.')

    return model, tokenizer


def T5_outputs_embedding(model, tokenizer, input_text):
    # Use T5 to get text embeddings.
    txt_list = [msg['value'][0] for conversation in input_text for msg in conversation if 'human' in msg['from']]
    human_texts = [''.join(txt_list)]

    inputs = tokenizer(human_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=inputs.input_ids)

    outputs_embedding = outputs.encoder_last_hidden_state  # torch.Size([1, 47, 1024])

    return outputs_embedding
    

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from'][0]
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'][0] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'][0] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'][0] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Llama') != -1:
                pass
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0,
                 scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.scale_factor = scale_factor 
    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=False)

class InpaintGenerator(BaseNetwork):
    def __init__(self, args):
        super(InpaintGenerator, self).__init__()
        # encoder
        self.encoder = Encoder(
            size = 512,
            style_dim = 8,
            channel_multiplier=2,
            narrow=1,
            device='cuda')

        self.decoder = Decoder(
            size = 512,
            style_dim = 8,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            isconcat=True,
            narrow=1,
            device='cuda'
        )

        self.vrt = Stage(in_dim=512,
                         dim=512,
                         input_resolution=(6, 64, 64),
                         depth=8,
                         num_heads=8,
                         window_size=[6, 16, 16],
                         mul_attn_ratio=0.75,
                         mlp_ratio=2.,
                         qkv_bias=True,
                         qk_scale=None,
                         drop_path=0.,
                         norm_layer=nn.LayerNorm,
                         use_checkpoint_attn=False,
                         use_checkpoint_ffn=False)

        self.final_conv = ConvLayer(512, 4096, 3, downsample=True)
        self.final_linear = EqualLinear(32*32, 1, activation='fused_lrelu')
        self.norm = nn.LayerNorm(512)
        self.attention_feat = attention()
        self.conv_512 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.conv_256 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.conv_128 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.back_512 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            nn.Conv2d(256, 256, 1, stride=1, padding=0), nn.LeakyReLU(),
            deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 1, stride=1, padding=0), nn.LeakyReLU(),
            deconv(128, 64, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
        )
        self.back_256 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            nn.Conv2d(256, 256, 1, stride=1, padding=0), nn.LeakyReLU(),
            deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU()
        )
        self.back_128 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU()
        )
        self.back_65 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.back_64 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.back_63 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
                                  deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
                                  deconv(128, 64, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
                                  nn.Conv2d(64, 1, 3, 1, 1), nn.LeakyReLU())
        self.prompt_learner = PromptLearner()
        self.device = torch.cuda.current_device()
        self.init_weights()
                
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        print(f'Initializing language decoder from {vicuna_ckpt_path} ...')
        self.prompt_control = prompt_control()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        # from peft import PeftModel
        self.llama_model = AutoModelForCausalLM.from_pretrained(vicuna_ckpt_path, torch_dtype=torch.float32, output_hidden_states=True)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.requires_grad_(False)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print('Language decoder initialized.')

        
    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding=None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2(54)
        # s1(4)
        batch_size = img_embeds.size(0)
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
        # employ T5
        # p_after_outputs = self.T5_model(input_ids)
        # p_after_embeds = p_after_outputs.last_hidden_state.expand(batch_size, -1, -1)
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim

        if anomaly_embedding != None:
            anomaly_embedding = anomaly_embedding.reshape(batch_size, -1, 4096)
            inputs_embeds = torch.cat(
                [bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds],
                dim=1)  # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size,
                            1 + p_before_embeds.size()[1] + 1 + p_middle_embeds.size()[1] + anomaly_embedding.size()[
                                1]],  # 1 (bos) + s1 + 1 (image vector)
                           dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1 + 1)

            targets = torch.cat([empty_targets, target_ids], dim=1)  # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1 + p_middle_embeds.size()[1] +
                                      anomaly_embedding.size()[1]], dtype=torch.long).to(
                self.device)  # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask

    def gen_acc(self, logits, targets):
        chosen_tokens = torch.max(logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return gen_acc

    def forward(self, source_tensor, output_texts):
        # print(f"output_texts:{output_texts}")
        decoder_noise = self.encoder(source_tensor)
        style = self.final_conv(decoder_noise[-1])
        style = self.final_linear(style.view(style.shape[0], 4096, -1)).reshape(style.shape[0], -1, 4096)
        vrt = torch.stack([self.conv_512(decoder_noise[0]),
                           self.conv_256(decoder_noise[1]),
                           self.conv_128(decoder_noise[2]),
                           decoder_noise[3], decoder_noise[4], decoder_noise[5]], dim=2)
        B, C, T, H, W = vrt.shape
        attention_feat, attentions = self.attention_feat(source_tensor, vrt.view(B*T, C, H, W), output_texts)
        anomaly_map_prompts = self.prompt_learner(attention_feat.view(B, C*T, H, W)) # B*T, 18, 4096
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, 1024)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(style, input_ids, target_ids, attention_mask, anomaly_map_prompts)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds.to(torch.float32),
            attention_mask=attention_mask.to(torch.float32),
            return_dict=True,
            labels=targets)
        gen_acc = self.gen_acc(outputs.logits, targets)
        # to calculate imperfection loss
        
        # the transformer part
        alpha, beta, gamma = self.prompt_control(outputs.hidden_states[-1])
        vrt = self.vrt(vrt.reshape(B, C, T, H, W), attention_feat.reshape(B, C, T, H, W), alpha.reshape(B, C, T, H, W), beta.reshape(B, C, T, H, W), gamma.reshape(B, C, T, H, W)).reshape(B, T, H, W, C)
        vrt = self.norm(vrt).reshape(T, B, C, H, W)
        vrt = vrt.reshape(T, B, C, H, W)
        attention_feat = attention_feat.reshape(T, B, C, H, W)

        # to calculate imperfection loss
        for atten in attention_feat:
            atten = self.conv(atten)
            attentions.append(atten)
        # decoder part
        feature = [512, 256, 128, 65, 64, 63]
        for i in range(6):
            block = getattr(self, f"back_{feature[i]}")
            # raw_noise.append((decoder_noise[i] * block(1-attention_feat[i])) - block(vrt[i] * attention_feat[i]))
            decoder_noise[i] = (decoder_noise[i] * block(1-attention_feat[i])) + block(vrt[i] * attention_feat[i])
        result = self.decoder(decoder_noise[::-1])
        return result, outputs, attentions, gen_acc




# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=False):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32
        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(5, 5),
                          stride=(2, 2),
                          padding=(1, 1),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 1,
                          nf * 2,
                          kernel_size=(5, 5),
                          stride=(2, 2),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 2,
                          nf * 4,
                          kernel_size=(5, 5),
                          stride=(2, 2),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 4,
                          nf * 8,
                          kernel_size=(5, 5),
                          stride=(1, 1),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 8,
                          nf * 16,
                          kernel_size=(5, 5),
                          stride=(1, 1),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 16,
                      nf * 16,
                      kernel_size=(5, 5),
                      stride=(1, 1),
                      padding=(2, 2)))
        self.init_weights()
    def forward(self, xs, mid_feat=None):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        # xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs)
        # print(feat.shape)
        # print(mid_feat.shape)
        if mid_feat != None:
            feat = feat * mid_feat
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        # out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return feat


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)



class img_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))

    def forward(self, x, img):
        sigmoid_params = self.convs(img) # torch.Size([1, 2, res, res])
        alpha, beta = torch.split(sigmoid_params, 512, dim=1)
        return x*alpha + beta

####################################
## Supervised Attention Module #####
class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True))
        self.feat_attention = img_attention()

        T5_model_path = "/share/home/HCI/dingchun/RetouchGPT/Flan-T5-Large"
        print(f"Initialing T5 model from {T5_model_path} ...")
        
        self.T5_model, self.T5_tokenizer = T5_model_load(T5_model_path)
        self.txt_linear = nn.Linear(1024, 512)
        self.txt_expand = nn.Linear(64, 64 * 64)
        self.txt_conv = nn.Sequential(
            nn.Conv2d(512, 512, 1, padding=0, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, padding=0, stride=1), nn.LeakyReLU(0.2, inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, padding=0, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 512, 1, padding=0, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True))


    def forward(self, img, feat, txt):
        # print(f"txt:{txt}\n")
        img_atten = self.conv0(feat)
        b, c, h, w = img_atten.shape
        txt_atten = T5_outputs_embedding(self.T5_model, self.T5_tokenizer, txt).permute(0, 2, 1)
        B, C, L = txt_atten.shape
        txt_atten = self.txt_linear(txt_atten.reshape(B, L, C)).reshape(B, 512, L)
        if L > 64:
            txt_atten = txt_atten[:, :, :64]
        else:
            padding_size = 64 - L
            padding = torch.zeros((B, 512, padding_size)).to(txt_atten.device)
            txt_atten = torch.cat((txt_atten, padding), dim=2)
        txt_atten = self.txt_expand(txt_atten)
        txt_atten = txt_atten.repeat(b//B, 1, 1)
        txt_atten = txt_atten.reshape(b, c, h, w)
        txt_atten = self.txt_conv(txt_atten)
        # print(f"img_atten shape: {img_atten.shape}")
        # print(f"txt_atten shape: {txt_atten.shape}")
        atten = torch.cat([img_atten, txt_atten], dim=1)
        atten = self.conv1(atten)
        atten = self.feat_attention(atten, img)
        return torch.sigmoid(atten), []

####################################

class PromptLearner(nn.Module):
    def __init__(self, dim_in=512, dim_out=4096):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in * 6, dim_in * 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(dim_in * 2, dim_in, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(dim_in * 4, dim_out, kernel_size=4, padding=0, stride=2),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)

    def forward(self, input):
        B, C, H, W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B, 4096, 9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output


class prompt_control(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 512)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512 * 2, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 2, 512 * 4, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 4, 512 * 6, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 6, 512 * 6, 1, padding=0, stride=1))
        
    def forward(self, x):
        # atten = self.linear(x)
        B, C, L = x.shape
        if C > 256:
            x = x[:, :256, :]
        else:
            padding_size = 256 - C
            padding = torch.zeros((B, padding_size, L)).to(x.device)
            x = torch.cat((x, padding), dim=1)
        x = self.linear(x.reshape(B, L, 256))
        x = x.reshape((B, 512, 64, 64))
        x = self.conv(x)
        return torch.sigmoid(x)

class prompt_control(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 512)
        self.conv0 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512 * 2, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 2, 512 * 4, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 4, 512 * 6, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 6, 512 * 12, 1, padding=0, stride=1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512 * 2, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 2, 512 * 4, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 4, 512 * 6, 3, padding=1, stride=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512 * 6, 512 * 6, 1, padding=0, stride=1))
        
    def forward(self, x):
        # atten = self.linear(x)
        B, C, L = x.shape
        if C > 256:
            x = x[:, :256, :]
        else:
            padding_size = 256 - C
            padding = torch.zeros((B, padding_size, L)).to(x.device)
            x = torch.cat((x, padding), dim=1)
        x = self.linear(x.reshape(B, L, 256))
        x = x.reshape((B, 512, 64, 64))
        sigma = self.conv0(x)
        gamma = self.conv1(x)
        alpha, beta = torch.split(sigma, 512 * 6, dim=1)
        return alpha, beta, gamma
