from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv
import cv2

from pdb import set_trace as breakpoint

# good solution !!!!

# Set the paths of the datasets here.
_CIFAR_DATASET_DIR = './datasets/CIFAR'
#_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'
_PLACES205_DATASET_DIR = './datasets/Places205'
#_IMAGENET_DATASET_DIR = '../imagenet/ILSVRC/Data/CLS-LOC'
#_IMAGENET_DATASET_DIR = '/home/rggadde/efs/rggadde/data/imagenet/ILSVRC/Data/CLS-LOC'
#_IMAGENET_DATASET_DIR = '/home/medathati/Work/SpectralSelfSupervision/Data/ILSVRC/Data/CLS-LOC'
#_IMAGENET_DATASET_DIR = '/home/medathati/Work/SpectralSelfSupervision/Data/tiny-imagenet-200' # This is tiny Imagenet
#_IMAGENET_DATASET_DIR = '/root/Data/tiny-imagenet-200'
_IMAGENET_DATASET_DIR = '/root/Data/ILSVRC/Data/CLS-LOC'

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(long(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name=='imagenet':
            assert(self.split=='train' or self.split=='val')
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.split!='train':
                transforms_list = [
                    transforms.Scale(256),
                    transforms.CenterCrop(224), # 224
                    lambda x: np.asarray(x),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomSizedCrop(224), #224
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Scale(256),#256
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _IMAGENET_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)
        elif self.dataset_name=='places205':
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            if self.split!='train':
                transforms_list = [
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
            self.transform = transforms.Compose(transforms_list)
            self.data = Places205(root=_PLACES205_DATASET_DIR, split=self.split,
                transform=self.transform)
        elif self.dataset_name=='cifar10':
            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _CIFAR_DATASET_DIR, train=self.split=='train',
                download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))

        if num_imgs_per_cat is not None:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)


    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))

        if self.dataset_name=='cifar10':
            labels = self.data.test_labels if (self.split=='test') else self.data.train_labels
            data = self.data.test_data if (self.split=='test') else self.data.train_data
            label2ind = buildLabelIndex(labels)
            all_indices = []
            for cat in label2ind.keys():
                label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
                all_indices += label2ind[cat]
            all_indices = sorted(all_indices)
            data = data[all_indices]
            labels = [labels[idx] for idx in all_indices]
            if self.split=='test':
                self.data.test_labels = labels
                self.data.test_data = data
            else:
                self.data.train_labels = labels
                self.data.train_data = data

            label2ind = buildLabelIndex(labels)
            for k, v in label2ind.items():
                assert(len(v)==num_imgs_per_cat)

        elif self.dataset_name=='imagenet':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(dname))
        elif self.dataset_name=='place205':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(dname))
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))


    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def split_bands(img, num_bands=4):
    assert(num_bands > 1)
    rows, cols, channels = img.shape
    crow, ccol = rows/2 , cols/2 # center
    #Divide into uniform bands.
    rspace = crow - np.linspace(0,crow,num_bands)
    cspace = ccol - np.linspace(0,ccol,num_bands)
    imgs = [np.float32(img)]
    for i in range(1, num_bands):
        # create a mask first, center square is 1, remaining all zeros
        mask1 = np.zeros((rows, cols, 2), np.float32)
        mask2 = np.zeros((rows, cols, 2), np.float32)
        mask1[int(crow-rspace[i-1]):int(crow+rspace[i-1]), int(ccol-cspace[i-1]):int(ccol+cspace[i-1]),:] = 1
        mask2[int(crow-rspace[i  ]):int(crow+rspace[i  ]), int(ccol-cspace[i  ]):int(ccol+cspace[i  ]),:] = 1
        filtered_img = np.zeros(img.shape)
        
        for c in range(channels):
            f = cv2.dft(np.float32(img[:,:,c]), flags=cv2.DFT_COMPLEX_OUTPUT)
            #f = np.fft.fft2(np.float32(img[:,:,c]))
            f_shift = np.fft.fftshift(f)
        
            ##Original image reconstruction
            #f_ishift = np.fft.ifftshift(f_shift)
            #filtered_img[:,:,c] = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            ##filtered_img[:,:,c] = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
            #f_ishift = np.fft.ifftshift(f_shift * (mask2[:,:,0] - mask1[:,:,1]))
            #temp = np.fft.ifft2(f_ishift)
            f_ishift = np.fft.ifftshift(f_shift * (mask2 - mask1))
            #filtered_img[:,:,c] = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            img_back = cv2.idft(f_ishift)
            filtered_img[:,:,c] = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        imgs.append(np.float32(filtered_img))
    return imgs

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis \
            else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis \
            else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

@torch.no_grad()
def MinMaxNormalize(X):
    X_channel_flat = X.view(*(X.size()[:-2]),1,-1)
    X_channel_min,_ = torch.min(X_channel_flat,len(X.size())-1, keepdim=True, out=None)
    X_channel_max,_ = torch.max(X_channel_flat,len(X.size())-1, keepdim=True, out=None)
    X_channel_den = X_channel_max - X_channel_min
    X_channel_den[X_channel_den==0] = 1.0 # To avoid division by zero
    X_normalized_flat = (X_channel_flat - X_channel_min)/X_channel_den
    X_normalized = X_normalized_flat.view(X.size())
    return X_normalized


@torch.no_grad()
def root_filter(img,num_filters=2):
    assert(num_filters>1)
    cuda = torch.device('cuda')
    img_cu = torch.from_numpy(img.transpose([2,0,1])).float().to('cpu')
    img_cu  = MinMaxNormalize(img_cu)
    imgs = [img]
    img_cu.unsqueeze_(0)
    I_fft = torch.rfft(img_cu, signal_ndim=2, onesided = False, normalized=False)
    I_mag = ((I_fft[:,:,:,:,0]**2+I_fft[:,:,:,:,1]**2)**0.5)
    #I_mag_nth = I_mag**(1-0.1)
    pf = 1.0/num_filters
    I_mag_nth = I_mag**(pf)
    for i in range(num_filters):
        I_fft[:,:,:,:,0] = I_fft[:,:,:,:,0]/I_mag_nth
        I_fft[:,:,:,:,1] = I_fft[:,:,:,:,1]/I_mag_nth
        I_fft[I_fft!=I_fft]=0
        I_hat = torch.irfft(I_fft, signal_ndim=2, onesided = False, normalized=False)
        I_hat_normalized = MinMaxNormalize(I_hat)
        I_hat_normalized = I_hat_normalized.cpu().numpy()[0].transpose([1,2,0])
        imgs.append(I_hat_normalized)
    return imgs

@torch.no_grad()
def split_bands_torch(img, num_bands=4):
    assert(num_bands > 1)
    imgs = [img]
    I = torch.from_numpy(img.transpose([2,0,1])).float().to('cpu')
    #I = transforms.ToTensor()(np.array(img.transpose([2,0,1])))
    I.unsqueeze_(0)
    I_fft = torch.rfft(I, signal_ndim=2, onesided=False, normalized=False)
    I_shift = batch_fftshift2d(I_fft)

    _, _, rows, cols, _ = I_shift.shape
    crow, ccol = rows/2 , cols/2     # center
    rspace = crow - np.linspace(0, crow, num_bands)
    cspace = ccol - np.linspace(0, ccol, num_bands)

    for i in range(1, num_bands):
        mask1 = torch.zeros(I_shift.shape) #(rows, cols, 2), np.uint8)
        mask2 = torch.zeros(I_shift.shape) #(rows, cols, 2), np.uint8)
        mask1[:, :, int(crow-rspace[i-1]):int(crow+rspace[i-1]),
              int(ccol-cspace[i-1]):int(ccol+cspace[i-1]), :] = 1
        mask2[:, :, int(crow-rspace[i  ]):int(crow+rspace[i  ]),
              int(ccol-cspace[i  ]):int(ccol+cspace[i  ]), :] = 1

        I_ishift = batch_ifftshift2d(I_shift * (mask2 - mask1))
        I_ifft = torch.ifft(I_ishift, signal_ndim=2, normalized=False)
        I_back = torch.sqrt(I_ifft[:,:,:,:,0]**2 + I_ifft[:,:,:,:,1]**2)
        I_back = I_back.cpu().numpy()[0].transpose([1,2,0])
        imgs.append(I_back)
    return imgs

# Source: https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

import pytorch_colors as colors
def shift_hue_cpu(arr,hshift):
    #print("Rotating hue by ",hshift*360)
    #arr = torch.from_numpy(arr.copy())
    #arr = torch.from_numpy(arr.transpose([2,0,1]).copy()).float()
    #arr.unsqueeze_(0)
    #print("The shape before hue rotate:", arr.size())
    hsv=rgb_to_hsv(arr)
    #hsv[...,0]=hshift #To set the hue
    hsv[...,0]= (hsv[...,0] + hshift)%1.0
    rgb=hsv_to_rgb(hsv)
    #print("The shape after hue rotate:", rgb.size())
    return rgb
@torch.no_grad()
def shift_hue(arr,hshift):
    #print("Rotating hue by ",hshift*360)
    #arr = torch.from_numpy(arr.copy())
    arr = torch.from_numpy(arr.transpose([2,0,1]).copy()).float()
    arr.unsqueeze_(0)
    #print("The shape before hue rotate:", arr.size())
    hsv=colors.rgb_to_hsv(arr)
    #print("Shape of HSV:", hsv.size())
    #hsv[...,0]=hshift #To set the hue
    #hsv[...,0]= (hsv[...,0] + hshift)%1.0
    hsv[:,0,...]= (hsv[:,0,...] + hshift)%1.0
    rgb=colors.hsv_to_rgb(hsv)
    #print("The shape after hue rotate:", rgb.size())
    return rgb.numpy()[0].transpose([1,2,0])

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                num_bands =  4

                # #filtered_imgs = split_bands(img0, num_bands=num_bands)
                # #print("This function is calling the split_bands_torch during training")
                # filtered_imgs = split_bands_torch(img0, num_bands=num_bands)
                # #filtered_imgs =  filtered_imgs.cpu().numpy()
                # filtered_imgs =[self.transform(img) for img in filtered_imgs]
                # filtered_labels = torch.arange(0, num_bands) # torch.LongTensor([0, 1, 2, 3])
                # return torch.stack(filtered_imgs, dim=0), filtered_labels

                #num_bands =  3
                #filtered_imgs = root_filter(img0, num_filters=num_bands)
                #filtered_labels = torch.arange(0, num_bands+1) 
                #filtered_imgs =  filtered_imgs.cpu().numpy()
                #filtered_imgs =[self.transform(img) for img in filtered_imgs]
                #return torch.stack(filtered_imgs, dim=0), filtered_labels
                
                #rotated_imgs = [
                #    self.transform(img0),
                #    self.transform(rotate_img(img0,  90).copy()),
                #    self.transform(rotate_img(img0, 180).copy()),
                #    self.transform(rotate_img(img0, 270).copy())
                #]
                #rotation_labels = torch.LongTensor([0, 1, 2, 3])
                #print("Label size from data laoder:", rotation_labels.size())
                #return torch.stack(rotated_imgs, dim=0), rotation_labels, rotation_labels

                #Hue Rotated Images
                #rotated_imgs = [
                #     self.transform(img0),
                #     self.transform(shift_hue(img0,  90/360.0).copy()),
                #     self.transform(shift_hue(img0, 180/360.0).copy()),
                #     self.transform(shift_hue(img0, 270/360.0).copy())
                #]
                #rotation_labels = torch.LongTensor([0, 1, 2, 3])
                #return torch.stack(rotated_imgs, dim=0), rotation_labels
                #Geometric and Photometric Rotated Images
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90).copy()),
                    self.transform(rotate_img(img0, 180).copy()),
                    self.transform(rotate_img(img0, 270).copy()),
                    self.transform(shift_hue(img0,  90/360.0).copy()),
                    self.transform(shift_hue(img0, 180/360.0).copy()),
                    self.transform(shift_hue(img0, 270/360.0).copy()),
                    self.transform(rotate_img(shift_hue(img0,  90/360.0),  90).copy()),
                    self.transform(rotate_img(shift_hue(img0, 180/360.0),  90).copy()),
                    self.transform(rotate_img(shift_hue(img0, 270/360.0),  90).copy()),
                    self.transform(rotate_img(shift_hue(img0,  90/360.0), 180).copy()),
                    self.transform(rotate_img(shift_hue(img0, 180/360.0), 180).copy()),
                    self.transform(rotate_img(shift_hue(img0, 270/360.0), 180).copy()),
                    self.transform(rotate_img(shift_hue(img0,  90/360.0), 270).copy()),
                    self.transform(rotate_img(shift_hue(img0, 180/360.0), 270).copy()),
                    self.transform(rotate_img(shift_hue(img0, 270/360.0), 270).copy())
                ]
                geo_rotation_labels = torch.LongTensor([0, 1, 2, 3,0,0,0,1,1,1,2,2,2,3,3,3])
                hue_rotation_labels = torch.LongTensor([0, 0, 0, 0,1,2,3,1,2,3,1,2,3,1,2,3])
                return torch.stack(rotated_imgs, dim=0), geo_rotation_labels, hue_rotation_labels

                # rotated_imgs = [
                #     self.transform(img0),
                #     self.transform(shift_hue(img0,  45/360.0).copy()),
                #     self.transform(shift_hue(img0, 90/360.0).copy()),
                #     self.transform(shift_hue(img0, 135/360.0).copy()),
                #     self.transform(shift_hue(img0, 180/360.0).copy()),
                #     self.transform(shift_hue(img0, 225/360.0).copy()),
                #     self.transform(shift_hue(img0, 270/360.0).copy()),
                #     self.transform(shift_hue(img0, 315/360.0).copy())
                # ]
                # rotation_labels = torch.LongTensor([0, 1, 2, 3,4,5,6,7])
                # return torch.stack(rotated_imgs, dim=0), rotation_labels

            def _collate_fun(batch):
                batch = default_collate(batch)
                #print("Total batch size:", len(batch))
                assert(len(batch)==3)
                batch_size, rotations, channels, height, width = batch[0].size()
                #print("batch_Size:", batch_size, "Rotations", rotations, "channlels", channels, height, width)
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                batch[2] = batch[2].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    #dataset = GenericDataset('cifar10','train', random_sized_crop=False)
    dataset = GenericDataset('imagenet','train', random_sized_crop=True)
    dataloader = DataLoader(dataset, batch_size=4, unsupervised=False)

    for b in dataloader(0):
        data, label = b
        print(label)
        break
    print(data.size())
    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0)/4,4,i+1)
        fig=plt.imshow(inv_transform(data[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
