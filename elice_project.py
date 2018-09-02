import os, time
from elice_solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

args = dict()
args['c_dim'] = 5
args['c2_dim'] = 8
args['celeba_crop_size'] = 178
args['image_size'] = 128
args['g_conv_dim'] = 64
args['d_conv_dim'] = 64
args['g_repeat_num'] = 6
args['d_repeat_num'] = 6
args['lambda_cls'] = 1
args['lambda_rec'] = 10
args['lambda_gp'] = 10
# Training configuration.
args['dataset'] = 'CelebA'
args['batch_size'] = 16
args['num_iters'] = 200000
args['num_iters_decay'] = 100000
args['g_lr'] = 0.0001
args['d_lr'] = 0.0001
args['n_critic'] = 5
args['beta1']=0.5
args['beta2']=0.999
args['resume_iters']=None
args['selected_attrs']=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
# Test configuration.
args['test_iters'] = 200000

#Miscellaneous
args['num_workers'] = 1
args['mode'] = 'test'
#Directories
args['celeba_image_dir'] = 'test_data/'
args['attr_path'] = 'test_data/info.txt'
args['log_dir'] = 'stargan/logs'
args['model_save_dir'] = 'stargan_celeba_128/models'
args['sample_dir'] = 'stargan/samples'
args['result_dir'] = 'test_data/result'
# Step size.
args['log_step'] = 10
args['sample_step'] = 1000
args['model_save_step'] = 10000
args['lr_update_step'] = 1000
config = args

celeba_loader = get_loader(config['celeba_image_dir'], config['attr_path'], config['selected_attrs'],
                           config['celeba_crop_size'], config['image_size'], config['batch_size'],
                           'CelebA', config['mode'], config['num_workers'])

start = time.time()
solver = Solver(celeba_loader, config)
solver.test()
end = time.time()
print(end - start)
