import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

plt.ion()

"""
    This module is designed for demonstrate the performance of pretrained models visually.
"""


def load_model(model_path, model_template) -> nn.Module:
    """
        :param model_path:
        :parameter model_template is the class of your models
    """
    loaded_model = model_template()
    loaded_model.load_state_dict(torch.load(model_path))
    return loaded_model


def data_processor(data_path, data_transforms=None, transforms_method=None, ) -> list:
    """
        receive a transforms method and data path(or a list of data path)
        :return will return a list of tensor(4 dimension, usually be (1,3,h,w) for rgb or (1,1,h,w) for grey-scale map )
        if data_path just have one picture this function will return a tensor of image
        if data_path is a list this function will return a list of tensor

        if you have some particular requests (such as not only using torchvision.transforms)
        you could assign a method 'transforms_method' to process  your image data
    """
    data_list = []
    if transforms_method is not None:
        data_transforms = transforms_method

    if isinstance(data_path, list):
        for data_i in data_path:
            data_list.append(data_transforms(Image.open(data_i)).unsqueeze(dim=0))
    else:
        data_single = Image.open(data_path)
        return data_transforms(data_single).unsqueeze(dim=0)
    return data_list


def show_image_by_plot(image_tensor, gray=False, sigmoid=False) -> None:
    """
        show image by using matplotlib, but this function just implement for single image.
        gray = True the plt will show a grey-scale map
    """
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor.squeeze(dim=0)
    out = image_tensor.permute(1, 2, 0)
    if sigmoid:
        out = torch.sigmoid(out)
    out = out.detach().numpy()
    if gray:
        plt.imshow(out, cmap='gray')
    else:
        plt.imshow(out)
    plt.show()


def model_infer(model, inputs):
    """
        receive a list of inputs and models to infer
        :returns is a tuple of outputs or a tensor
    """
    outputs = model(*inputs)
    # print(outputs.shape)
    # if isinstance(outputs, tuple):
    #     return outputs
    # else:
    #     return tuple(outputs)
    return outputs


'''
How to use it:
'''
'''
RGBD
'''
# if __name__ == '__main__':
#     def test_method(image):
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor()
#         ])
#         image = transform(image)
#         print(image.size())
#         if image.size()[0] == 1:
#             image = torch.cat((image, image, image), dim=0)
#         return image
#
#
#     net = load_model('../saved_models/baseLineModel/Baseline_1_gen.pth', model_template=VGG16Baseline).eval()
#
#     data = data_processor(['../test/000008_left.jpg', '../test/000008_left.png'],
#                           transforms_method=test_method)
#     side_out5, side_out4, side_out3, side_out2, side_out1 = model_infer(net, data)
#     print(len(side_out1.size()))
#     show_image_by_plot(side_out1, gray=True, sigmoid=True)
'''
rgbt
'''
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize([352,352]),
        transforms.ToTensor()
    ])


    net = load_model('../saved_models/baseLineModel_RGBT/BaselineRGBT_100_gen.pth', model_template=VGG16Baseline).eval()

    data = data_processor(['../test/33-rgb.jpg', '../test/33-t.jpg'],
                          data_transforms=transform)
    side_out5, side_out4, side_out3, side_out2, side_out1 = model_infer(net, data)
    print(len(side_out1.size()))
    show_image_by_plot(side_out1, gray=True, sigmoid=True)
