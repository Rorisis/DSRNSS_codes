###### Modified from: https://github.com/vikolss/DACS  ###########
import random

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    # if param['road'] == None:
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    # else:
    #     data, target = one_mix_1(mask=param['mix'], data=data, target=target, road_mask = param['road'])
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=img.device).view(1, 3, 1, 1)
    return img.mul(std).add(mean)


def renorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=img.device).view(1, 3, 1, 1)
    return img.sub(mean).div(std)


def color_jitter(color_jitter, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                # NOTE: this assumes images are normalized
                data = denorm(data)
                data = seq(data)
                data = renorm(data)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        # print("label:",label.shape)
        # NOTE: this seems to be a bug, we keep it for consistency
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        # print("class_masks:", class_masks)
        # print("class_masks.shape:", class_masks.shape)
    return class_masks

# def get_dynamic_masks(labels, pseudo_label):
#     dynamic_class_masks = [5,6,7,11,12,13,14,15,16,17,18]
#     road_class = [0]
#     road_class = torch.tensor(road_class).cuda()
#     road_masks = []
#     class_masks = []

#     # print("labels:",labels.shape)
    
#     for i in range(labels.shape[0]):
#         spatial_matrix = mask_to_onehot(labels[i].cpu().numpy(), 19)
#         # torch.norm(spatial_matrix)
#         argmax_u_w = torch.mul(torch.tensor(spatial_matrix).cuda(), pseudo_label[i].unsqueeze(0)) # torch.Size([1, 19, 512, 512])
#         argmax_u_w = argmax_u_w.max(1).indices.squeeze(0)  # torch.Size([1, 512, 512]) --> torch.Size([512, 512])
#         # NOTE: this seems to be a bug, we keep it for consistency
#         classes = torch.unique(argmax_u_w)
#         nclasses = classes.shape[0]
#         classes = torch.Tensor([value for value in dynamic_class_masks if value in classes]).cuda()
#         # print("classes",classes)
#         # class_choice = np.random.choice(
#         #     nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
#         # classes = classes[torch.Tensor(class_choice).long()].cuda()
#         # print("argmax_u_w:",argmax_u_w.shape)
#         class_masks.append(generate_class_mask(argmax_u_w, classes).unsqueeze(0))
#         road_masks.append(generate_class_mask(pseudo_label[i], road_class).unsqueeze(0))
#         # class_masks.append(generate_class_mask(labels[i], classes).unsqueeze(0))
#         # print("class_masks:", len(class_masks))
#     return class_masks, road_masks

def get_dynamic_masks(labels, pseudo_label):
    class_masks = []
    for i in range(labels.shape[0]):
        # spatial_matrix = mask_to_onehot(labels[i].cpu().numpy(), 19)
#         # torch.norm(spatial_matrix)
        # argmax_u_w = torch.mul(torch.tensor(spatial_matrix).cuda(), pseudo_label[i].unsqueeze(0)) # torch.Size([1, 19, 512, 512])
        # argmax_u_w = argmax_u_w.max(1).indices.squeeze(0)  # torch.Size([1, 512, 512]) --> torch.Size([512, 512])
        classes = torch.unique(labels[i])
        class_list = classes.cpu().numpy()
        nclasses = classes.shape[0]
        categories_index = np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)
        categories = class_list[categories_index]
        categories_new = categories
        #----------------------------------------------------------------
        # Group 1 of meta class: object
        # if contains traffic light, cut the pole
        if categories_new.__contains__(6) and class_list.__contains__(5):
            categories_new = np.unique(np.append(categories_new, 5))
            # print('judge 6, append 5:', categories)
        # if contains traffic sign, cut the pole
        if categories_new.__contains__(7) and class_list.__contains__(5):
            categories_new = np.unique(np.append(categories_new, 5))
            # print('judge 7, append 5:', categories)
        # if contains pole, cut the traffic sign and traffic light
        if categories_new.__contains__(5):
            if class_list.__contains__(6):
                categories_new = np.unique(np.append(categories_new, 6))
                # print('judge 5, append 6:', categories)
            if class_list.__contains__(7):
                categories_new = np.unique(np.append(categories_new, 7))
                # print('judge 5, append 7:', categories)
        #----------------------------------------------------------------
        # Group 2 of meta class: human-vehicle
        # if contains rider, cut the bicycle and motorcycle 
        if categories_new.__contains__(12):
            if class_list.__contains__(18):
                categories_new = np.unique(np.append(categories_new, 18))
            if class_list.__contains__(17):
                categories_new = np.unique(np.append(categories_new, 17))
        classes_rt = (torch.Tensor(categories_new).long()).cuda()
        classes_drop = classes_rt
        class_masks.append(generate_class_mask(labels[i], classes_drop).unsqueeze(0))
    return class_masks

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def generate_class_mask(label, classes):
    # 类的mask
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask 


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target

def one_mix_1(mask, data=None, target=None, road_mask = None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        stackedMask1, _ = torch.broadcast_tensors(road_mask[0], data[0])
        if torch.nonzero(stackedMask1) > 0.3 * data[1].shape[-1]:
            print("1:",torch.nonzero(stackedMask1))
            print("2:",stackedMask0.shape)
            diff =int(torch.nonzero(stackedMask1)[-1] - torch.nonzero(stackedMask0)[-1]) #上方
            if  diff > 30:
                stackedMask0 = nn.functional.pad(stackedMask0[:, 0:(stackedMask0.shape[-1]-diff)], (diff, 0), 'constant', 0)
            # elif diff < -30:
            #     stackedMask0 = nn.functional.pad(stackedMask0[:, 0:(stackedMask0.shape[-1]-diff)], (diff, 0), 'constant', 0)
            data = (stackedMask0 * data[0] +
                    (1 - stackedMask0) * data[1]).unsqueeze(0)
        else:
            data = (stackedMask0 * data[0] +
                    (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        stackedMask1, _ = torch.broadcast_tensors(road_mask[0], data[0])
        if torch.nonzero(stackedMask1) > 0.3 * data[1].shape[-1]:
            print("1:",torch.nonzero(stackedMask1))
            print("2:",stackedMask0.shape)
            diff =int(torch.nonzero(stackedMask1)[-1] - torch.nonzero(stackedMask0)[-1], data[0]) #上方
            if  diff > 30:
                print(stackedMask0[0:(stackedMask0.shape[-1]-diff),:])
                stackedMask0 = torch.cat(torch.zero(diff, (stackedMask0.shape[-1]-diff)), stackedMask0[0:(stackedMask0.shape[-1]-diff),:], dim = 0)
            elif diff < -30:
                stackedMask0 = torch.cat(stackedMask0[diff:(stackedMask0.shape[-1]),:], torch.zero(0, diff), dim = 0)
            target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
        else:
            target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
        
    return data, target
