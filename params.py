import torch
import random
import numpy as np
import logging
import os

#############################
# DataSet Path
#############################
#@markdown Training dataset
soundlabel = "Meow" #@param ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_Bass","Drawer_open_or_close","Electric_piano", "Fart", "Finger_snapping","Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica","Hi-hat","Keys_jangling","Knock","Laughter", "Meow","Microwave_oven","Oboe", "Saxophone","Scissors", "Shatter","Snare_drum", "Squeak","Tambourine", "Tearing", "Telephone","Tearing", "Telephone", "Trumpet", "Violin_or_fiddle","Writing"]
target_signals_dir = "/content/drive/MyDrive/TFG/datasets/" + soundlabel

#############################
# Model Params
#############################
#@markdown Name of the model to be saved
model_prefix = "meow_verified_4_s_400k_16khz" #@param {type:"string"}
output_dir = "/content/drive/MyDrive/TFG/SFXGan/training_process/meow4s_16khz"#@param {type:"string"}
#@markdown ---
n_iterations = 400000 #@param {type:"number"}
lr_g = 1e-5 #@param {type:"number"}
lr_d = 3e-5  #@param {type:"number"}# you can use with discriminator having a larger learning rate than generator instead of using n_critic updates ttur https://arxiv.org/abs/1706.08500
#@markdown ---
beta1 = 0.5
beta2 = 0.9
use_batchnorm=False
validate = True #@param ["True", "False"] {type:"raw"}
decay_lr = False # used to linearly deay learning rate untill reaching 0 at iteration 100,000
generator_batch_size_factor = 1 # in some cases we might try to update the generator with double batch size used in the discriminator https://arxiv.org/abs/1706.08500
n_critic = 1 # update generator every n_critic steps if lr_g = lr_d the n_critic's default value is 5
# gradient penalty regularization factor.
p_coeff = 10
batch_size = 20#@param {type:"number"}
noise_latent_dim = 100  # size of the sampling noise
model_capacity_size = 32    # model capacity during training can be reduced to 32 for larger window length of 2 seconds and 4 seconds
# rate of storing validation and costs params
store_cost_every = 1000#@param {type:"number"}
progress_bar_step_iter_size = 1000# @param {type:"number"}
#############################
# Backup Params
#############################
take_backup = True
backup_every_n_iters = 1000#@param {type:"number"}
save_samples_every = 5000#@param {type:"number"}
if not(os.path.isdir(output_dir)):
    os.makedirs(output_dir)
#############################
# Audio Reading Params
#############################
window_length = 65536 #@param ["16384", "32768", "65536"] {type:"raw"}
sampling_rate = 16000 #@param {type:"number"}
normalize_audio = True
num_channels = 1

#############################
# Logger init
#############################
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)
#############################
# Torch Init and seed setting
#############################
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# update the seed
manual_seed = 2019
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
if cuda:
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()
