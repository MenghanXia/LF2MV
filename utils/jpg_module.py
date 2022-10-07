import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
from ._dct import dct_2d, idct_2d
import torch

class Quantize(Function):
    # quantize with 255 level, input range [0,1]
    # demo use:
    #   quantize = Quantize.apply
    #   res = quantize(inputs)
    @staticmethod
    def forward(ctx, input_):
        # ctx.save_for_backward(input_)
        # output = torch.clamp(input_, min=0, max=1)
        # output *= 255.
        output = torch.round(input_)
        # output /= 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        return grad_input

class JPEG_Layer(nn.Module):
    def __init__(self, quality=85, norm='ortho'):
        super(JPEG_Layer, self).__init__()
        self.quality = quality
        self.normal = norm
        self.quantize = Quantize.apply
        self.patchsize = 8
        self.QmatY, self.QmatCrCb = self.getQuantizeMat(jpg_quality=quality)
        self.QmatY = self.quantize(self.QmatY)
        self.QmatCrCb = self.quantize(self.QmatCrCb)
        self.lvl = 255

    def forward(self, img):
        # img is [-1.,1.] to [0.,1.]
        img = img * 0.5 + 0.5
        imsize = img.shape
        if self.QmatY.data.type() != img.data.type():
            if img.is_cuda:
                self.QmatY = self.QmatY.cuda(img.get_device())
                self.QmatCrCb = self.QmatCrCb.cuda(img.get_device())
            self.QmatY = self.QmatY.type_as(img)
            self.QmatCrCb = self.QmatCrCb.type_as(img)
        # rgb to ycbcr
        img255 = img * self.lvl
        if imsize[1] == 3:
            img255 = self.rgb2ycbcr(img255)
            img255 = self.quantize(img255)
        # image to 8x8 blocks
        img_unf = F.unfold(img255, kernel_size=(self.patchsize, self.patchsize), stride=(self.patchsize, self.patchsize))
        unf_shape = img_unf.shape
        img_unf = img_unf.view([unf_shape[0], imsize[1], self.patchsize, self.patchsize, unf_shape[2]])
        # dct
        img_unf = img_unf.transpose(2, 4).contiguous()
        img_unf -= 128
        img_unf = dct_2d(img_unf, norm=self.normal)
        # quantization with matrix
        Qmat_list = [self.QmatY, self.QmatCrCb, self.QmatCrCb]
        for ii in range(img_unf.shape[1]):
            img_unf[:, ii, :, :, :] = self.quantizeBlocks(img_unf[:, ii, :, :, :], Qmat_list[ii])
        img_unf = self.quantize(img_unf)
        # inverse quantization with matrix
        for ii in range(img_unf.shape[1]):
            img_unf[:, ii, :, :, :] = self.quantizeBlocks(img_unf[:, ii, :, :, :], 1 / Qmat_list[ii])
        # idct
        img_unf = idct_2d(img_unf, norm=self.normal)
        img_unf += 128
        img_unf = img_unf.transpose(2, 4).contiguous()
        # 8x8 blocks to iamge
        img_unf = img_unf.view(unf_shape)
        img_out = F.fold(img_unf, output_size=imsize[2:], kernel_size=(self.patchsize, self.patchsize),
                         stride=(self.patchsize, self.patchsize))
        if imsize[1] == 3:
            img_out = self.ycbcr2rgb(img_out)
            img_out = self.quantize(img_out)
        img_out /= self.lvl
        # img is [0.,1.] to [-1.,1.]
        img_out = img_out * 2.0 - 1.0
        return img_out

    @staticmethod
    def quantizeBlocks(blocks, Qmat):
        blocks_shape = blocks.shape
        blocks = blocks.contiguous().view(-1, blocks_shape[-2], blocks_shape[-1])
        for ii in range(blocks.shape[-2]):
            for jj in range(blocks_shape[-1]):
                blocks[:, ii, jj] /= Qmat[ii, jj]
        blocks = blocks.view(blocks_shape)
        return blocks

    @staticmethod
    def rgb2ycbcr(im):
        # https://en.wikipedia.org/wiki/YCbCr
        r = im[:, 0, :, :]
        g = im[:, 1, :, :]
        b = im[:, 2, :, :]
        # Y
        Y = 0.256788 * r + 0.504129 * g + 0.0979059 * b + 16.
        # Cb
        Cb = 128 - .148224 * r - .290992 * g + .439216 * b
        # Cr
        Cr = 128 + .439216 * r - .367788 * g - .0714275 * b
        if len(Y.shape) == 3:
            Y = Y.view([Y.shape[0], -1, Y.shape[-2], Y.shape[-1]])
            Cb = Cb.view([Cb.shape[0], -1, Cb.shape[-2], Cb.shape[-1]])
            Cr = Cr.view([Cr.shape[0], -1, Cr.shape[-2], Cr.shape[-1]])
        im_out = torch.cat([Y, Cb, Cr], 1)
        return im_out

    @staticmethod
    def ycbcr2rgb(im):
        # https://en.wikipedia.org/wiki/YCbCr
        y = im[:, 0, :, :] - 16
        cb = im[:, 1, :, :] - 128
        cr = im[:, 2, :, :] - 128
        # R
        r = 1.16438 * y + 3.01124e-7 * cb + 1.59603 * cr
        # G
        g = 1.16438 * y - .391763 * cb - .812968 * cr
        # B
        b = 1.16438 * y + 2.01723 * cb + 0.00000305426 * cr
        if len(r.shape) == 3:
            r = r.view([r.shape[0], -1, r.shape[-2], r.shape[-1]])
            g = g.view([g.shape[0], -1, g.shape[-2], g.shape[-1]])
            b = b.view([b.shape[0], -1, b.shape[-2], b.shape[-1]])
        im_out = torch.cat([r, g, b], 1)
        return im_out

    # get quantization matrix for Y and CrCb
    @staticmethod
    def getQuantizeMat(jpg_quality):
        # quality is 0-100
        mat50Y = torch.Tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])
        mat50CrCb = torch.Tensor([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 66, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]])

        if jpg_quality < 1:
            jpg_quality = 1
        if jpg_quality > 100:
            jpg_quality = 100
        if jpg_quality < 50:
            scale_factor = 5000. / jpg_quality
        else:
            scale_factor = 200 - jpg_quality * 2.
        matY = (mat50Y * scale_factor + 50) / 100
        matY = torch.clamp(matY, min=1, max=255)
        matCrCb = (mat50CrCb * scale_factor + 50) / 100
        matCrCb = torch.clamp(matCrCb, min=1, max=255)
        return matY, matCrCb

