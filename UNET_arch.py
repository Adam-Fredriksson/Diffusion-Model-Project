import torch
import torch.nn as nn
import math
import os

DROPOUT = 0.1

def calc_pixel_size(size, kernel, padding, stride, num_convs=2, down=True, layer_num=0):
    """Helper function for calculating the pixel size at a specific location in the model."""
    for i in range(num_convs):
        size = (size-kernel+2*padding)/stride + 1
    if down and layer_num>0:
        return size//2
    elif layer_num>0 and not down:
        return size * 2
    else:
        return size 

 
class SinusoidalPositionEmbeddings(nn.Module):
    """Code inspired from: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=HhIgGq3za0yh"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, tstep,max_period=10000):
          half_dim = self.dim // 2
          exp = -math.log(max_period) / (half_dim - 1)
          embeddings = torch.exp(exp * torch.linspace(0, 1, half_dim, device=tstep.device))
          embeddings = tstep.float() * embeddings
          embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=1)
          return embeddings
    

class ClassEmbedding(nn.Module):
    """Legacy code, replaced by torch.nn.Embedding."""
    def __init__(self, dim, onehot=False) -> None:
        super().__init__()
        self.dim = dim
        self.layer = nn.Linear(1,self.dim) if not onehot else nn.Identity()
        
    def forward(self, c):
        c = self.layer(c)
        return c


class FastAttention(nn.Module):

    def __init__(self, channels, n_head):
        super().__init__()
        self.channels = channels
        self.n_heads = n_head

    def forward(self, query, key, value):
        chunk_size = query.shape[-1] // self.n_heads
        query = torch.split(query, chunk_size, dim=-1)
        key = torch.split(key, chunk_size, dim=-1)
        value = torch.split(value, chunk_size, dim=-1)
        attn_val = None
        for chunk_idx in range(self.n_heads):
            temp = nn.functional.scaled_dot_product_attention(query[chunk_idx], key[chunk_idx], value=value[chunk_idx], dropout_p=DROPOUT)
            if attn_val is None:
                attn_val = temp
            else:
                attn_val = torch.concat([attn_val, temp], dim=-1)
        return attn_val
            

class AttentionBlock(nn.Module): 
    """Code initially inspired by https://github.com/dome272/Diffusion-Models-pytorch/tree/main, but the main thing remaining is self.fcl which is legacy code and only
    used to load earlier models. Models used in report does not use self.fcl."""
    def __init__(self, channels, size, n_head=4, fast_attention=False):
        """Channels is the number of channels that the images has and size is the number of pixels (along one direction, i.e. a 32x32 images
        has size=32)."""
        super().__init__()
        self.channels = channels
        self.size = int(size)
        if fast_attention:
            self.fast_attn = True
            self.attn = FastAttention(channels=channels, n_head=n_head)
        else:
            self.fast_attn = False
            self.attn = nn.MultiheadAttention(channels, n_head, batch_first=True, dropout=DROPOUT)
        self.group_norm = nn.GroupNorm(8,channels)
        
        #NOTE: self.fcl is LEGACY but trained models contain it, though they do not use it!
        self.fcl = nn.Sequential(
            # nn.GroupNorm(1,channels),
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels))
        #NOTE: self.fcl is LEGACY but trained models contain it, though they do not use it!

    def forward(self, x): 
        group_norm_x = self.group_norm(x)
        
        group_norm_x = group_norm_x.view(-1, self.channels, self.size*self.size)
        group_norm_x = group_norm_x.swapaxes(1,2)
        
        if self.fast_attn:
            attn_val = self.attn(group_norm_x, group_norm_x, group_norm_x)
        else:
            attn_val, _ = self.attn(group_norm_x, group_norm_x, group_norm_x)
        attn_val = attn_val.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
        attn_val = attn_val + x
        return attn_val
    

class CNNBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, t_emdim=32,class_emdim=32, kernel=3, padding=1, device='cpu') -> None:
        super().__init__()
        self.time_layer = nn.Linear(t_emdim, out_channels)
        self.class_layer = nn.Linear(class_emdim, out_channels)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.SiLU())
        self.GN = nn.GroupNorm(8, in_channels) if in_channels % 8 == 0 else nn.GroupNorm(1, in_channels).to(device)
        self.GN_out = nn.GroupNorm(8, out_channels) if out_channels % 8 == 0 else nn.GroupNorm(1, out_channels).to(device)
        self.conv_layer_final = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, padding=padding),
            nn.SiLU())
        self.conv_layer_final = self.conv_layer_final.to(device)
        self.activation = nn.SiLU() 
        if not in_channels == out_channels:
            self.residual_conv_layer = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x, t, c=None):
        t = self.time_layer(t)[:, :, None, None]
        if c is not None:
            c = self.class_layer(c)[:, :, None, None]
        res = x
        # res = x.clone()
        x = self.activation(self.GN(x)) #Model12 did not have activation here, only GN
        x = self.conv_layer(x)
        
        final_input = x + t + c if c is not None else x + t
        if res.shape[1] == x.shape[1]:
            return self.GN_out(self.conv_layer_final(final_input)) + res
        else:
            temp = self.residual_conv_layer(res)
            return self.GN_out(self.conv_layer_final(final_input) + temp)


class UpSampleCNNBlock(nn.Module):

    def __init__(self, in_channels=64, out_channels=3, t_emdim=32, class_emdim=32, n_head=4, use_cross_attn=False):
        super().__init__()
        self.conv_layer = CNNBlock(in_channels=2*out_channels, out_channels=out_channels, t_emdim=t_emdim, class_emdim=class_emdim)
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, (2,2), 2)
        self.cross_attention = nn.MultiheadAttention(out_channels, n_head, batch_first=True, dropout=DROPOUT)
        self.cross_attn_GN = nn.GroupNorm(8,out_channels)
        self.use_cross_attn = use_cross_attn

    def forward(self, x, res, t, c=None):
        x = self.upsample(x)
        B, C, W, H = x.shape
        crop_W = (res.shape[2] - W) // 2
        crop_H = (res.shape[3] - H) // 2
        
        
        if crop_W > 0 and crop_H > 0:
            res = res[:, :, crop_W:-crop_W, crop_H:-crop_H]
        
        if self.use_cross_attn:
            normed_x = self.cross_attn_GN(x)
            normed_x = normed_x.reshape(-1, C, H*W).swapaxes(1,2)
            normed_res = self.cross_attn_GN(res)
            normed_res = normed_res.reshape(-1,C,H*W).swapaxes(1,2)
            attn_val,_ = self.cross_attention(normed_x,normed_res, normed_res)
            attn_val = attn_val.swapaxes(2,1).reshape(-1,C,W,H)
        x = torch.cat([x, res], dim=1)
        if self.use_cross_attn:
            return self.conv_layer(x, t, c) + attn_val
        return self.conv_layer(x, t, c)
    

class MidBlock(nn.Module):

    def __init__(self,channel, pixel_size, t_emdim=32, fast_attn=False):
        super().__init__()
        self.GN = nn.GroupNorm(8, channel) if channel%8==0 else nn.GroupNorm(1, channel)
        self.act = nn.SiLU()
        self.block1 = CNNBlock(channel,channel)
        self.attn = AttentionBlock(channels=channel, size=pixel_size, fast_attention=fast_attn)
        self.block2 = CNNBlock(channel, channel)
        self.dropout = nn.Dropout(DROPOUT)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emdim),
            nn.Linear(t_emdim, t_emdim),
            nn.SiLU()
            )
        self.class_mlp = nn.Embedding(10, t_emdim)
    
    def forward(self, x, t, c=None):
        t = self.time_mlp(t)
        if c is not None:
            c = self.class_mlp(c)
            c = c.view(-1, c.shape[-1])
        x = self.act(self.GN(x))
        x = self.block1(x,t,c)
        x = self.attn(x)
        x = self.dropout(x)
        x = self.block2(x,t,c)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channel_list, output_size, t_emdim,class_emdim=32,class_onehot=False, pixel_dims = 32, device='cpu', fast_attn=False):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn_block = nn.ModuleList()
        self.num_blocks = len(in_channel_list)
        self.dropout = nn.Dropout(DROPOUT)
        pixel_sizes = [pixel_dims]
        self.GN = nn.GroupNorm(1, in_channel_list[0])
        self.input_layer = nn.Conv2d(in_channels=in_channel_list[0], out_channels=in_channel_list[1], kernel_size=3, padding=1)
        for i in range(self.num_blocks-1):
            layer = CNNBlock(in_channel_list[i], out_channels=in_channel_list[i+1], t_emdim=t_emdim, class_emdim=class_emdim)
            self.block.append(layer)
            layer = CNNBlock(in_channel_list[i+1], out_channels=in_channel_list[i+1], t_emdim=t_emdim, class_emdim=class_emdim)
            self.block.append(layer)
            new_size = calc_pixel_size(pixel_sizes[-1], kernel=3, padding=1, stride=1, layer_num=i)
            attn_layer = AttentionBlock(in_channel_list[i+1], new_size, fast_attention=fast_attn)
            pixel_sizes.append(new_size)
            self.attn_block.append(attn_layer)
        self.block.append(CNNBlock(in_channels=in_channel_list[-1], out_channels=output_size, t_emdim=t_emdim, class_emdim=class_emdim))
        self.block.append(CNNBlock(in_channels=output_size, out_channels=output_size, t_emdim=t_emdim, class_emdim=class_emdim))
        self.pool = nn.MaxPool2d(2)
        # print(len(self.block))
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emdim),
            nn.Linear(t_emdim, t_emdim),
            nn.SiLU()
            )
        self.class_mlp = nn.Embedding(10, class_emdim)
        
        self.block = self.block.to(device)
        self.class_mlp = self.class_mlp.to(device)
        self.time_mlp = self.time_mlp.to(device)
        self.attn_block = self.attn_block.to(device)
        
        
    def forward(self, x, t, c=None):
        residuals = []
        x = self.GN(x)
        t = self.time_mlp(t)
        if c is not None:
            c = self.class_mlp(c)
            c = c.view(-1, c.shape[-1])
        for i in range(self.num_blocks):
            B, C, W, H = x.shape
            x = self.block[2*i](x, t, c)
            x = self.dropout(x)
            if i < self.num_blocks - 1:
                x = self.attn_block[i](x)
            x = self.block[2*i+1](x,t, c)
            if not i == self.num_blocks - 1:
                residuals.append(x)
                x = self.pool(x)

        return x, residuals
    

class Decoder(nn.Module):

    def __init__(self, in_channel_list, output_size, t_emdim,class_emdim=32,class_onehot=False,
                  use_diffusion=False, pixel_dims=4, use_cross_attn=False, device='cpu', fast_attn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.block = nn.ModuleList()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emdim),
            nn.Linear(t_emdim, t_emdim),
            nn.SiLU()
            )
        self.class_mlp = nn.Embedding(10, class_emdim)
        
        pixel_sizes = [pixel_dims]
        self.num_blocks = len(in_channel_list)
        self.dropout = nn.Dropout(DROPOUT)
        self.attn_block = nn.ModuleList()
        for i in range(len(in_channel_list)-1):
            layer = UpSampleCNNBlock(in_channel_list[i], out_channels=in_channel_list[i+1], t_emdim=t_emdim, class_emdim=class_emdim, use_cross_attn=use_cross_attn)
            self.block.append(layer)
            layer = CNNBlock(in_channel_list[i+1], out_channels=in_channel_list[i+1], t_emdim=t_emdim, class_emdim=class_emdim)
            self.block.append(layer)
            new_size = calc_pixel_size(pixel_sizes[-1], kernel=3, padding=1, stride=1, down=False, layer_num=i+1)
            attn_layer = AttentionBlock(in_channel_list[i+1], new_size, fast_attention=fast_attn)
            pixel_sizes.append(new_size)
            self.attn_block.append(attn_layer)
        layer = CNNBlock(in_channel_list[i+1], out_channels=in_channel_list[i+1], t_emdim=t_emdim, class_emdim=class_emdim)
        self.block.append(layer)
        layer = CNNBlock(in_channel_list[i+1], out_channels=output_size, t_emdim=t_emdim, class_emdim=class_emdim)
        
        self.block.append(layer)
        self.block = self.block.to(device)
        self.class_mlp = self.class_mlp.to(device)
        self.time_mlp = self.time_mlp.to(device)
        self.attn_block = self.attn_block.to(device)
        
        if use_diffusion:
            self.output_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel_list[-1], out_channels=output_size, kernel_size=(2,2), stride=2),
                                               nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=1))
        else:
            self.output_layer = CNNBlock(in_channels=in_channel_list[-1], out_channels=output_size,t_emdim=t_emdim, class_emdim=class_emdim, kernel=1, padding=0).to(device)
            

    def forward(self, x, residuals, t, c=None):
        num_residuals = len(residuals)
        t = self.time_mlp(t)
        if c is not None:
            c = self.class_mlp(c)
            c = c.view(-1, c.shape[-1])
        for i in range(self.num_blocks):
            if i < self.num_blocks - 1:
                x = self.block[2*i](x, residuals[num_residuals - 1 -i], t, c)
                x = self.attn_block[i](x)
                x = self.dropout(x)
            else:
                x = self.block[2*i](x, t, c)
                
            x = self.block[2*i + 1](x, t, c)
        # x = self.attn_block[-1](x)
        # x = self.output_layer(x, t, c)
        return x

    
class U_Net(nn.Module):
  """Create a U-Net architecture. The number of resolution levels will be the length of the encoder channel list (enc_channel_list) minus one, since the first
  element specifies the input channel. The dec_channel_list should be the enc_channel_list in reverse. Embedding sizes t_emdim and class_emdim should in general be the same
  and pixel_dims_enc is the starting pixel size of the image. If pixel_dims_enc is None, then the pixel size at the bottom of the model will be calculated. Resolution is halved
  each level."""
  def __init__(self,enc_channel_list, dec_channel_list, t_emdim, class_emdim=32, pixel_dims_enc=128, pixel_dims_dec=None,
                use_cross_attn=False, class_onehot=False, device='cpu', use_mid_blocks=True, fast_attn=False):
    super().__init__()
    if pixel_dims_dec is None:
        pixel_dims_dec = pixel_dims_enc // 2**(len(enc_channel_list)-2)

    self.encoder = Encoder(enc_channel_list[:-1], enc_channel_list[-1], t_emdim,  class_emdim=class_emdim, pixel_dims=pixel_dims_enc, 
                           class_onehot=class_onehot, device=device, fast_attn=fast_attn)
    self.encoder = self.encoder.to(device)
    self.decoder = Decoder(dec_channel_list[:-1], dec_channel_list[-1], t_emdim, class_emdim=class_emdim, pixel_dims=pixel_dims_dec,
                            use_cross_attn=use_cross_attn, class_onehot=class_onehot, device=device, fast_attn=fast_attn)
    self.decoder = self.decoder.to(device)
    self.use_mid_blocks = use_mid_blocks
    if use_mid_blocks:
        final_pixel_size = calc_pixel_size(self.encoder.attn_block[-1].size, 3, 1, stride=1, num_convs=1, layer_num=len(enc_channel_list))
        self.mid_blocks = MidBlock(enc_channel_list[-1], final_pixel_size, t_emdim, fast_attn=fast_attn)
  def forward(self, X, t, c=None):
    enc_out, res = self.encoder(X, t, c)
    if self.use_mid_blocks:
        enc_out = self.mid_blocks(enc_out, t, c)
    dec_out = self.decoder(enc_out, res, t, c)
    return dec_out
  

def save_only_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, model, optimizer=None, scheduler=None):
    """Function for loading model from file with path model_path. Input model should have the same architecture as the model being loaded, since loading a
    Pytorch model means updating the weights of the new, inputted model to be equal to the weights stored in the saved model file. Optionally an optimizer
    and scheduler can be loaded too, if they were saved during training (not by default)."""
    print('Loading Model...')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = None
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if 'save_interval' in checkpoint:
        save_interval = checkpoint['save_interval']
    else:
        save_interval = None
    
    # If you want to load optimizer state and other information, add the following lines
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model,optimizer,scheduler, loss, epoch, save_interval


def load_all(model, optimizer, scheduler, dir, device):
    """Legacy loading function."""
    if os.path.exists(dir+"/model.pth"):
        model.load_state_dict(torch.load(dir + "/model.pth", map_location=torch.device(device)))
        optimizer.load_state_dict(torch.load(dir + "/optimizer.pth", map_location=torch.device(device)))
        scheduler.load_state_dict(torch.load(dir + "/scheduler.pth", map_location=torch.device(device)))
        #cosineScheduler.load_state_dict(torch.load(dir + "/cos_scheduler.pth", map_location=torch.device('cpu')))
        #warmUpScheduler.load_state_dict(torch.load(dir + "/warmup_scheduler.pth", map_location=torch.device('cpu')))
        #trainer.load_state_dict(torch.load(dir + "/trainer.pth", map_location=torch.device('cpu')))

        print("loaded all model, etc.")
    return model, optimizer, scheduler


def load_only_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        # print("loaded model")
    return model



