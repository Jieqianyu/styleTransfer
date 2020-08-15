import os
import torch
from PIL import Image
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from libs.utils import print_options
from libs.models import encoder3,encoder4, encoder5
from libs.models import decoder3,decoder4, decoder5
# import matplotlib.pyplot as plt

class Opt(object):
  def __init__(self):
    self.vgg_dir = 'models/vgg_r31.pth' # pre-trained encoder path
    self.decoder_dir = 'models/dec_r31.pth' # pre-trained decoder path
    self.matrix_dir = "models/r31.pth" # path to pre-trained model
    self.style = "data/style/rain_princess.jpg" # path to style image
    self.content = "data/content/chicago.png" # path to content image
    self.loadSize = 512 # scale image size
    self.layer = "r31" # features of which layer to transform
    self.outf = "output/" # output folder

class Transfer(object):
    def __init__(self, opt=None, load_deafult=False):
        # PREPARATIONS
        if opt:
            self.opt = opt
        else:
            self.opt = Opt()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_options(self.opt)

        os.makedirs(self.opt.outf, exist_ok=True)
        cudnn.benchmark = True

        self.load_model()
        if load_deafult:
            self.load_data(self.opt.style, self.opt.content)
        

    def load_model(self):
        # MODEL
        if(self.opt.layer == 'r31'):
            self.vgg = encoder3()
            self.dec = decoder3()
        elif(self.opt.layer == 'r41'):
            self.vgg = encoder4()
            self.dec = decoder4()
        self.matrix = MulLayer(layer=self.opt.layer)

        self.vgg.load_state_dict(torch.load(self.opt.vgg_dir))
        self.dec.load_state_dict(torch.load(self.opt.decoder_dir))
        self.matrix.load_state_dict(torch.load(self.opt.matrix_dir, map_location=self.device))
        self.vgg.to(self.device)
        self.dec.to(self.device)
        self.matrix.to(self.device)

    def load_data(self, style_img_path, content_img_path):
        transform = transforms.Compose([
                    transforms.Resize(self.opt.loadSize),
                    transforms.ToTensor()])

        self.styleV = transform(Image.open(style_img_path).convert('RGB')).unsqueeze(0).to(self.device)
        self.contentV = transform(Image.open(content_img_path).convert('RGB')).unsqueeze(0).to(self.device)

    def transfer(self):
        with torch.no_grad():
            sF = self.vgg(self.styleV)
            cF = self.vgg(self.contentV)

            if(self.opt.layer == 'r41'):
                feature,transmatrix = self.matrix(cF[self.opt.layer],sF[self.opt.layer])
            else:
                feature,transmatrix = self.matrix(cF,sF)
                transfer = self.dec(feature)

        transfer = transfer.clamp(0,1).squeeze(0).data.cpu().numpy()
        transfer = 255*transfer.transpose((1, 2, 0))
        # transfer = transfer.transpose((1,2,0))

        return transfer


if __name__ == '__main__':
    opt = Opt()
    opt.style = "static/style/rain_princess.jpg"
    opt.content = "static/content/chicago.png"
    contentName = os.path.splitext(os.path.basename(opt.content))[0]
    styleName = os.path.splitext(os.path.basename(opt.style))[0]
    transfer_obj = Transfer(opt)
    transfer = transfer_obj.transfer()
    # f, ax= plt.subplots(1, 1)
    # ax.axis('off')
    # ax.set_title('%s_stylized_%s.png'%(contentName,styleName))
    # plt.imshow(transfer)
    # plt.show()
