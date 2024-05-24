import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from UNET_arch import U_Net, save_only_model, load_model
import matplotlib.pyplot as plt
from copy import deepcopy

#Use your functions from the assignments to load the data!
# from main_diffusion import load_all_training_data, LoadBatch, preprocess

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomCIFAR10Dataset(Dataset):
    """[LEGACY]Custom Dataset class to convert Cifar10 numpy arrays to Pytorch dataset."""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target


def montage(W, is_batched=False, include_title_num=False):
	""" Display the image of each element in the list W."""
	num_plots = len(W)
	rows = 4 if num_plots % 4 == 0 else 2
	cols = num_plots //rows
	fig, ax = plt.subplots(rows, cols)
	
	for i in range(rows):
		for j in range(cols):
			temp = W[i*cols + j].cpu().detach().numpy()
			if is_batched or len(temp.shape) == 4:
				temp = temp[0]
			
			im = temp.transpose(1, 2, 0)  
			
			sim = (im - np.min(im)) / (np.max(im) - np.min(im))
			
			ax[i][j].imshow(sim)
			if include_title_num:
				ax[i][j].set_title("y=" + str(i * cols + j))
			ax[i][j].axis('off')
	
      
def train(data, betas, model, optimizer, epochs=10, alphas=None, file_name='model.pth', ema_copy=None, clip_value=1., use_class_cond=True, 
		  scheduler=None, last_epoch=0, losses=[], save_interval=50):
	
	model.train()
	if alphas is None:
		alphas = calc_alphas(betas)
	T = alphas.shape[0]
	supervised_loss_obj = torch.nn.MSELoss().to(DEVICE)	
	
	for epoch in range(epochs):
		counter = 0
		for x, y in tqdm(data):
			if use_class_cond:
				y = y.reshape(-1, 1).to(DEVICE)
				if torch.rand(1).item() < 0.1:
					y = None
			else:
				y = None
			
			x = x.to(DEVICE)

			counter += 1
			t = torch.randint(0, T-1, (x.shape[0],1), device=DEVICE)
			eps = torch.randn(x.shape, device=DEVICE)
			
			sqrt_alphas = torch.sqrt(alphas[t])
			sqrt_one_minus_alphas = torch.sqrt(1. - alphas[t])
			noisy_x = torch.einsum('ij,iklm->iklm',sqrt_alphas,x) + torch.einsum('ij,iklm->iklm',sqrt_one_minus_alphas,eps)
			
			model_output = model(noisy_x.to(torch.float32), t.to(torch.float32), y)
			loss = supervised_loss_obj(model_output, eps)
			if torch.isnan(loss):
				print('loss is nan')
				continue
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
			optimizer.step()
			losses.append(loss.item())
			scheduler.step()
			ema_copy = ema_param_update(model, ema_copy)
			# montage(noisy_x)
			if (counter+1)%save_interval == 0:
				torch.save({
							'epoch': epoch+last_epoch,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'scheduler_state_dict': scheduler.state_dict(),
							'loss': losses,
							'save_interval':save_interval
							}, file_name)
				torch.save({
							'epoch': epoch+last_epoch,
							'model_state_dict': ema_copy.state_dict(),
							'loss': losses,
							'save_interval':save_interval
							}, 'EMA_'+file_name)
				
		if (epoch+1)%1 == 0:
			print(f'Saving model, epoch number: {epoch +last_epoch + 1}')
			torch.save({
							'epoch': epoch+last_epoch+1,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'scheduler_state_dict': scheduler.state_dict(),
							'loss': losses,
							'save_interval':save_interval
							}, file_name)
			torch.save({
							'epoch': epoch+last_epoch+1,
							'model_state_dict': ema_copy.state_dict(),
							'loss': losses,
							'save_interval':save_interval
							}, 'EMA_'+file_name)
	return losses, ema_copy


@torch.no_grad()
def sample(model, betas,dimension=(1, 3, 32, 32),time_steps=1000, num_images=10, start_img=None, alphas=None, c=None, w_scale=0.1):
	"""Returns generated sample from model by starting from standard Gaussian noise."""
	x = torch.randn(size=dimension, device=DEVICE) if start_img is None else start_img
	if c is not None:
		c = torch.tensor(c, dtype=torch.int32, device=DEVICE).reshape(-1,1)
		
	if alphas is None:
		alphas = calc_alphas(betas)
	if start_img is not None:
		ind_t = torch.randint(time_steps-1, time_steps, (x.shape[0],1), device=DEVICE)
		
		eps = torch.randn(x.shape, device=DEVICE)

		sqrt_alphas = torch.sqrt(alphas[ind_t])
		sqrt_one_minus_alphas = torch.sqrt(1. - alphas[ind_t])
		x = torch.einsum('ij,iklm->iklm',sqrt_alphas,x) + torch.einsum('ij,iklm->iklm',sqrt_one_minus_alphas,eps)
	
	sigmas = torch.sqrt(betas)
	
	saved_images = [x]
	model.eval()
	for t in tqdm(range(time_steps-1, -1, -1)):
		noise = torch.randn(size=dimension, device=DEVICE) if t >= 1 else 0.
		t_step = torch.tensor(t, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(x.shape[0], -1)

		model_output = model(x.to(torch.float32), t_step, c)
		if w_scale > 0.:
			non_class_noise = model(x.to(torch.float32), t_step, None)
			# model_output = non_class_noise + w_scale*(model_output - non_class_noise)
			model_output = model_output + w_scale*(model_output - non_class_noise)
		
		x = (x - betas[t]*model_output / torch.sqrt(1 - alphas[t])) / torch.sqrt(1-betas[t]) + sigmas[t] * noise
		
		if (t) % (time_steps / (num_images-2)) == 0:
			saved_images.append(x)
		
	saved_images.append(x)
	return saved_images


def cosine_schedule(T):
	"""Cosine schedule from https://arxiv.org/abs/2102.09672"""
	s = torch.tensor(0.008) 
	f = lambda t: torch.min(torch.cos((t/T + s)*torch.pi/(2*(1+s)))**2, torch.tensor(0.9999))
	times = torch.arange(0, T)
	delta_t = times[1] - times[0]
	f_0 = f(0)
	alphas = f(times + delta_t) / f_0
	betas = torch.zeros_like(alphas)
	betas[0] = 1 - alphas[0] 
	for t in times[1:]:
		betas[t] = torch.min(torch.tensor(0.999), 1 - alphas[t]/alphas[t-1])
	
	return alphas.to(DEVICE), betas.to(DEVICE)


def test_alphas():
	T = 4000
	times = torch.arange(0, T)
	alphas_cos, betas_cos = cosine_schedule(T)
	print(alphas_cos[0])
	print(alphas_cos.min())
	print(alphas_cos.max())
	print(betas_cos.min())
	print(betas_cos.max())
	
	betas_lin = torch.linspace(1.e-4, 0.01, T)
	alphas_lin = calc_alphas(betas_lin)
	print(alphas_lin.min())
	print(alphas_lin.max())
	print(betas_lin.min())
	print(betas_lin.max())
	plt.plot(times / T, alphas_cos, label='COS')
	plt.plot(times/T, alphas_lin, label='LINEAR')
	plt.legend()
	plt.figure()
	plt.plot(times / T, betas_cos, label='COS')
	plt.plot(times/T, betas_lin, label='LINEAR')
	plt.legend()
	plt.show()


def calc_alphas(betas):
    alphas = torch.cumprod(torch.ones_like(betas) - betas, dim=0)
    return alphas
    

def apply_noise(x, alpha):
    x = x.unsqueeze(0)
    if x.dim() == 4:
        x = x.expand(alpha.shape[0], -1, -1, -1)
    else:
        x = x.expand(alpha.shape[0], -1, -1)
    eps = torch.randn(size=x.shape)
    dim_difference = x.dim() - alpha.dim()
    for _ in range(dim_difference):
        # alpha = alpha.reshape(alpha.shape[0], 1, 1)
        alpha = alpha.unsqueeze(-1)
    noisy_x = torch.sqrt(alpha)*x + torch.sqrt(1.-alpha)*eps
    return noisy_x, eps


def ema_param_update(model, ema_copy=None, decay=0.9999):
	"""EMA update of the main model."""
	if ema_copy is None:
		ema_copy = deepcopy(model).eval()
		for param in ema_copy.parameters():
			param.requires_grad_(False)
	else:
		ema_copy.eval()
	for ema_param, param in zip(ema_copy.parameters(), model.parameters()):
		ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
	
	return ema_copy


def load_data_set(batch_size, img_size, data_set_choice='mnist'):
	from torchvision import transforms, datasets
	transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),        
	])
	if data_set_choice=='mnist':
		dataset = datasets.MNIST(root='./mnist_data',
											train=True,      
										transform=transform, 
										download=True)  
	elif data_set_choice=='cifar':  
		dataset = datasets.CIFAR10(root='./cifar_data',
											train=True,      
										transform=transform, 
										download=True)  
	else:
		dataset = datasets.LFWPeople(root='./LFW_data',
                                    split='train',    
                                 transform=transform, 
                                 download=True)

	train_data = DataLoader(dataset=dataset,
							batch_size=batch_size, 
							shuffle=True)          
	return train_data

def load_cifar(batch_size):
	"""LEGACY"""
	def load_all_training_data(place_holder):
		return
	trainX, validX, trainOneHot, valid_one_hot,labelX, labelValid, testX, test_Y, test_one_hot = load_all_training_data(1)
	
	print(trainX.shape)
	
	train_dataset = CustomCIFAR10Dataset(trainX[:], labelX.T[:])
	# train_dataset = CustomCIFAR10Dataset(trainX[:-42000], trainOneHot.T[:-42000])
	valid_dataset = CustomCIFAR10Dataset(validX,valid_one_hot.T)
	test_dataset = CustomCIFAR10Dataset(testX, test_one_hot.T)
	train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
	test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	# montage(torch.tensor(testX)[:10])
	return train_data


def main():
	#-------------Create Variables, Optimizer, Scheduler and Model-------------
	print(DEVICE)
	use_cifar = True
	data_set_choice = 'cifar'
	file_name='model19.pth' if use_cifar else 'model3_MNIST.pth'
	time_steps = 1000
	epochs = 0
	last_epoch = 0
	batch_size = 24
	img_size =32
	channel_size = 3 if use_cifar else 1
	model = U_Net([channel_size,64,128,256, 512], [512,256,128,64,channel_size], 32, 32, pixel_dims_enc=img_size, use_cross_attn=True, device=DEVICE, use_mid_blocks=True, fast_attn=False)
	model = model.to(DEVICE)
	optim = torch.optim.AdamW(model.parameters(), lr=3.e-4)
	# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9999)
	T_max = int(2084 + 2084 / 2) if epochs%2==0 else 2084
	print(T_max)
	eta_min = 1.e-7
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=eta_min)
	losses = []
	
	#-------------Load Model, Optimizer and Scheduler-------------
	should_load=True 
	load_ema_model = True
	if should_load:
		# model, optim,scheduler,losses,last_epoch, save_interval = load_model('Models/model17_copy.pth', model, optimizer=optim, scheduler=scheduler)
		model, optim,scheduler,losses,last_epoch, save_interval = load_model(file_name, model, optimizer=optim, scheduler=scheduler)
		print(f'Epochs trained: {last_epoch}')
		
	if load_ema_model:
		ema_model = ema_param_update(model)
		ema_model, _, _,_,_,_ = load_model('EMA_'+file_name, ema_model)
	else:
		ema_model = None
	
	param_count = sum(p.numel() for p in model.parameters())
	param_count_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'Total: {param_count}')
	print(f'With grad: {param_count_grad}')

	#-------------Initialize Noise params-------------
	betas = torch.linspace(1.e-4, 0.02, time_steps, device=DEVICE)
	alphas = None	
	
	#-------------Load Data-------------
	train_data = load_data_set(batch_size, img_size=img_size, data_set_choice=data_set_choice)
	# if use_cifar:
	# 	train_data = load_cifar(batch_size)
	# else:
	# 	train_data = load_mnist(batch_size=batch_size, img_size=img_size)
	
	#-------------Training of Model-------------
	loss, ema_model = train(train_data, betas, model, optim, epochs=epochs, alphas=alphas, file_name=file_name, ema_copy=ema_model, 
						 scheduler=scheduler, last_epoch=last_epoch, losses=losses)
	
	#-------------Sampling and Plotting Images. Set Save Path and Weight Scale.-------------
	w_scale = 0.
	num_imgs_times_ten = 2
	sample_num = 10
	uncond = True
	c = None if uncond else list(range(10)) * num_imgs_times_ten
	print(c)
	# saved_img_file_name = 'Images/CIFAR_images/model19_epoch60_weight3_numImgs20_sample1.pth'
	# saved_EMA_img_file_name = 'Images/CIFAR_images/EMA_model19_epoch60_weight3_numImgs20_sample1.pth'
	saved_img_file_name = 'Images/CIFAR_images/model19_epoch'+str(last_epoch)+'_weight'+str(int(w_scale))+f'_numImgs{num_imgs_times_ten*10}_sample{sample_num}_uncond{uncond}.pth'
	saved_EMA_img_file_name = 'Images/CIFAR_images/EMA_model19_epoch'+str(last_epoch)+'_weight'+str(int(w_scale))+f'_numImgs{num_imgs_times_ten*10}_sample{sample_num}.pth'
	print(saved_img_file_name)
	should_save_model_img = True


	# img = sample(model, betas, dimension=(10, channel_size, 32,32),time_steps=time_steps, start_img=None, alphas=alphas, c=None, w_scale=0.)
	img = sample(model, betas, dimension=(10*num_imgs_times_ten, channel_size, 32,32),time_steps=time_steps, start_img=None, alphas=alphas,
			   c=c, w_scale=w_scale)
	if should_save_model_img:
		torch.save(img, saved_img_file_name)
	# img = torch.load(saved_img_file_name)
	montage(img[-1])
	
	if ema_model is None:
		ema_model = load_model('EMA_'+file_name, model)
	# # img = sample(ema_model, betas, dimension=(10, channel_size, 32,32),time_steps=time_steps, start_img=None, alphas=alphas, c=None, w_scale=0.)
	img = sample(ema_model, betas, dimension=(10*num_imgs_times_ten, channel_size, 32,32),time_steps=time_steps, start_img=None, alphas=alphas,
			   c=c, w_scale=w_scale)
	# img = torch.load(saved_EMA_img_file_name)
	montage(img[-1])
	if should_save_model_img:
		torch.save(img, saved_EMA_img_file_name)
	
	#-------------Loss Plot-------------
	plt.figure()
	plt.plot(np.arange(len(losses)), losses)
	plt.xlabel('Iteration', fontsize=20)
	plt.ylabel('Loss', fontsize=20)
	plt.show()
      

if __name__=='__main__':
	# test_alphas()
	main()