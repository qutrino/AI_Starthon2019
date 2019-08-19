from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import torch
import os
import numpy as np

def get_transform():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.Resize((224,224)))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)

def target_transform(target, sizes):
    width,height = sizes

    target[0] = float(target[0]) / float(width)
    target[1] = float(target[1]) / float(height)
    target[2] = float(target[2]) / float(width)
    target[3] = float(target[3]) / float(height)

    return target


class FlipOnly(object):
    """ Flip image target to [0,1]
    """
    def __init__(self, flip=0.5):
        self.flip = flip

    def __call__(self, sample):
        (pil_image, xywh) = sample # yet pil_image

        if np.random.uniform() < self.flip:
            image_w, image_h = pil_image.size

            x,y,w,h = xywh

            pil_image = transforms.functional.hflip(pil_image)
            xywh = [ image_w-x-w, y, w, h]

        return [ pil_image, xywh ]

class FlipAndCrop(object):
    """ Flip image target to [0,1]
    """
    def __init__(self, flip=0.5, crop=0.3):
        self.flip = flip
        self.crop = crop


    def __call__(self, sample):
        (pil_image, xywh) = sample # yet pil_image

        if np.random.uniform() < self.flip:
            image_w, image_h = pil_image.size

            x,y,w,h = xywh

            pil_image = transforms.functional.hflip(pil_image)
            xywh = [ image_w-x-w, y, w, h]

        if np.random.uniform() < self.crop:
            image_w, image_h = pil_image.size
            x,y,w,h = xywh

            m_x1 = x # left margin
            m_y1 = y

            m_x2 = image_w-x-w
            m_y2 = image_h-y-h

            new_m_x1 = np.random.randint(m_x1) if m_x1 >= 1 else 0
            new_m_y1 = np.random.randint(m_y1) if m_y1 >= 1 else 0
            new_m_x2 = np.random.randint(m_x2) if m_x2 >= 1 else 0
            new_m_y2 = np.random.randint(m_y2) if m_y2 >= 1 else 0

            new_image_w = image_w - (m_x1-new_m_x1) - (m_x2-new_m_x2)
            new_image_h = image_h - (m_y1-new_m_y1) - (m_y2-new_m_y2)

            pil_image = transforms.functional.crop(pil_image, new_m_y1, new_m_x1, new_image_h, new_image_w )       #y, x, h, w
            xywh = [ x-(m_x1-new_m_x1),\
                     y-(m_y1-new_m_y1),\
                     w, h ]

        return [ pil_image, xywh ]





class TargetNormalize(object):
    """ Normalize target to [0,1]
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        (pil_image, xywh) = sample # yet pil_image

        w, h = pil_image.size
        xywh_norm = [float(xywh[0]) / w, float(xywh[1]) / h, float(xywh[2]) / w, float(xywh[3]) / h]

        return [ pil_image, torch.FloatTensor(xywh_norm) ]

class ImageNormlize(object):
    def __init__(self, output_size):
        self.output_size = output_size

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform = []
        transform.append(transforms.Resize((output_size, output_size)))
        transform.append(transforms.ToTensor())
        transform.append(normalize)

        self.image_transform = transforms.Compose(transform)

    def __call__(self, sample):
        (pil_image, xywh_norm) = sample
        image = self.image_transform(pil_image)

        return [ image, xywh_norm ]

class ToHeatMap(object):
    """Add Heat"""
    def __init__(self, heat_size=128):
        self.heat_size = heat_size

    def __call__(self, sample):
        image, box = sample

        c, h, w = image.size()

        max_index = self.heat_size - 1

        new_box = [ box[0] * self.heat_size, \
                    box[1] * self.heat_size, \
                    (box[2]+box[0]) * self.heat_size, \
                    (box[3]+box[1]) * self.heat_size ]

        x1 = min(max_index,int(new_box[0]))
        y1 = min(max_index,int(new_box[1]))
        x2 = min(max_index,int(new_box[2]))
        y2 = min(max_index,int(new_box[3]))

        tl_map = torch.zeros([1, self.heat_size, self.heat_size], dtype=torch.float)
        br_map = torch.zeros([1, self.heat_size, self.heat_size], dtype=torch.float)

        tl_map[0, y1, x1] = 1
        br_map[0, y2, x2] = 1

        return [image, box , tl_map, br_map]


class ToHeatMapOffset(object):
    """Add Heat"""
    def __init__(self, heat_size=128):
        self.heat_size = heat_size

    def __call__(self, sample):
        image, box = sample

        c, h, w = image.size()

        max_index = self.heat_size - 1

        new_box = [ box[0] * self.heat_size, \
                    box[1] * self.heat_size, \
                    (box[2]+box[0]) * self.heat_size, \
                    (box[3]+box[1]) * self.heat_size ]

        x1 = min(max_index,int(new_box[0]))
        y1 = min(max_index,int(new_box[1]))
        x2 = min(max_index,int(new_box[2]))
        y2 = min(max_index,int(new_box[3]))

        dx1 = new_box[0] - x1
        dy1 = new_box[1] - y1
        dx2 = new_box[2] - x2
        dy2 = new_box[3] - y2


        tl_regrs = torch.zeros([1, 2], dtype=torch.float)
        br_regrs = torch.zeros([1, 2], dtype=torch.float)

        tl_regrs[0, 0] = dx1
        tl_regrs[0, 1] = dy1
        br_regrs[0, 0] = dx2
        br_regrs[0, 1] = dy2

        tl_map = torch.zeros([1, self.heat_size, self.heat_size], dtype=torch.float)
        br_map = torch.zeros([1, self.heat_size, self.heat_size], dtype=torch.float)

        tl_map[0, y1, x1] = 1
        br_map[0, y2, x2] = 1

        tl_tags     = np.zeros(( 1,), dtype=np.int64)
        br_tags     = np.zeros(( 1,), dtype=np.int64)
        tag_masks   = np.zeros(( 1,), dtype=np.uint8)

        tag_masks[0] = 1
        tl_tags[0] = y1 * self.heat_size + x1
        br_tags[0] = y2 * self.heat_size + x2
        tl_tags     = torch.from_numpy(tl_tags)
        br_tags     = torch.from_numpy(br_tags)
        tag_masks   = torch.from_numpy(tag_masks)


        return [image, box, tl_map, br_map, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags ]



class CustomDataset(data.Dataset):
    def __init__(self, root, transform, loader=default_loader):
        self.root = root
        self.transform = transform
        #self.target_transform = target_transform
        self.loader = loader

        self.images = []
        self.boxes = []
        dir_list = sorted(os.listdir(self.root))

        for file_path in dir_list:
            if file_path.endswith('.jpg'):
                self.images.append(os.path.join(self.root, file_path))

                box_file_path = file_path[:-4] + '.box'
                box_str = open(os.path.join(self.root, box_file_path), 'r', encoding='utf-8').read().split('\n')[0]
                box = [float(bb) for bb in box_str.split(',')]
                self.boxes.append(box)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pil_image = self.loader(self.images[index])
        #targets = torch.Tensor(self.boxes[index])
        whxy = self.boxes[index]


        return self.transform([pil_image, whxy])

        #inputs = self.transform(images)
        #width, height = images.size
        #targets = self.target_transform(targets, (width, height))


        #return inputs, targets

normal_transform = transforms.Compose( [ TargetNormalize(), ImageNormlize(511), ToHeatMapOffset(128)] )
aug_transform = transforms.Compose( [ FlipOnly(), TargetNormalize(), ImageNormlize(511),  ToHeatMapOffset(128)] )

def data_loader(root, phase='train', batch_size=10):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    dataset = CustomDataset(root, normal_transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)

# def data_loader_with_split(root, train_split=0.9, batch_size=10, val_label_file='./val_label'):
#     input_transform = get_transform()
#     dataset = CustomDataset(root, aug_transform)
#     split_size = int(len(dataset) * train_split)
#     train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])
#     tr_loader = data.DataLoader(dataset=train_set,
#                                 batch_size=batch_size,
#                                 shuffle=True)
#     val_loader = data.DataLoader(dataset=valid_set,
#                                  batch_size=batch_size,
#                                  shuffle=False)
#
#
#     gt_labels = [valid_set[idx][1] for idx in range(len(valid_set))]
#     gt_labels_string = [','.join([str(s.numpy()) for s in l])
#                         for l in list(gt_labels)]
#     with open(val_label_file, 'w') as file_writer:
#         file_writer.write("\n".join(gt_labels_string))
#
#     return tr_loader, val_loader, val_label_file


def data_loader_with_split(root, train_split=0.9, batch_size=10, val_label_file='./val_label'):
    input_transform = get_transform()
    aug_dataset = CustomDataset(root, aug_transform)
    normal_dataset = CustomDataset(root, normal_transform)

    split_size = int(len(normal_dataset) * train_split)

    indices = np.random.permutation(len(normal_dataset))
    training_idx, val_index = indices[:split_size], indices[split_size:]

    train_set = data.dataset.Subset(aug_dataset, training_idx)
    valid_set = data.dataset.Subset(normal_dataset, val_index)

    #train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])

    tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 shuffle=False)


    gt_labels = [valid_set[idx][1] for idx in range(len(valid_set))]
    gt_labels_string = [','.join([str(s.numpy()) for s in l])
                        for l in list(gt_labels)]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))

    return tr_loader, val_loader, val_label_file
