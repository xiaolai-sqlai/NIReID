import cv2
import random
import os
import os.path as osp

from . import functional as F


_cv2_interpolation_to_str = {
    cv2.INTER_NEAREST: 'INTER_NEAREST',
    cv2.INTER_LINEAR: 'INTER_LINEAR',
    cv2.INTER_AREA: 'INTER_AREA',
    cv2.INTER_CUBIC: 'INTER_CUBIC',
    cv2.INTER_LANCZOS4: 'INTER_LANCZOS4'
}


class CopyPaste(object):
    def __init__(self, probability=0.5, height_ratio=0.5, dataset_root="dev/shm", paste_dataset=None, interpolation=cv2.INTER_LINEAR):
        self.probability = probability
        self.height_ratio = height_ratio
        self.paste_dataset = paste_dataset
        self.interpolation = interpolation

        self.path_list = []
        if paste_dataset == "veri776":
            for path in ["image_train", "image_test", "image_query"]:
                for name in os.listdir(osp.join(dataset_root, "VeRi", path)):
                    self.path_list.append(osp.join(dataset_root, "VeRi", path, name))

        if paste_dataset == "vehicleid":
            for name in os.listdir(osp.join(dataset_root, "VehicleID_V1.0")):
                self.path_list.append(osp.join(dataset_root, "VehicleID_V1.0", name))

        self.len = len(self.path_list)

    def __call__(self, img):
        if self.len > 0:
            H, W = img.shape[:2]
            if random.uniform(0, 1) <= self.probability:
                x = random.randint(0, W - 1)
                y = random.randint(int(H * self.height_ratio), H - 1)

                vehicle_idx = random.randint(0, self.len - 1)
                vehicle_path = self.path_list[vehicle_idx]
                vehicle_img = F.imread(vehicle_path)
                resize_img = F.resize(vehicle_img, (W, W), self.interpolation)

                if y + W <= H:
                    img[y:y+W, x:W] = resize_img[:, :W-x]
                else:
                    img[y:H, x:W] = resize_img[:H-y, :W-x]

                # if H - y < W:
                #     resize_img = F.resize(vehicle_img, (W, W), self.interpolation)
                #     img[y:H, x:W] = resize_img[:H-y, :W-x]
                # else:
                #     resize_img = F.resize(vehicle_img, (H - y, W), self.interpolation)
                #     img[y:H, x:W] = resize_img[:, :W-x]
        return img

    def __repr__(self):
        interpolate_str = _cv2_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '({0}, paste dataset={1}, probability={2}, height ratio={3}, interpolation={4})'.format(self.len, self.paste_dataset, self.probability, self.height_ratio, interpolate_str)
