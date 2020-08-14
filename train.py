from deeplearning.backbone_pytorch.backbone.vgg import *
from deeplearning.backbone_pytorch.backbone.Shufflenet import *
from deeplearning.backbone_pytorch.backbone.Squeezenet import *
from deeplearning.backbone_pytorch.backbone.preactresnet import *
from deeplearning.backbone_pytorch.backbone.shufflenetV2 import *
from deeplearning.backbone_pytorch.utils import *
import torch.optim as optim

class Train:
    def __init__(self,data_path,epoch,lr):
        self.path = data_path
        self.epoch = epoch
        self.lr = lr
    def train(self):
        # net = VGG(make_layers(cfg['A'], batch_norm=True))
        # net = ShuffleNet([4, 8, 4])
        # net = SqueezeNet(class_num=10)
        # net = PreactResnet([3,4,23,3])
        net = ShuffleNetV2()
        trainloder, _ = Dataloder(path=self.path).dataset()
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        net.train()
        for i in range(self.epoch):
            for batch_index, (images, labels) in enumerate(trainloder):
                labels = labels
                images = images
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                print(loss.item())
            print("第%d轮的训练已经完成"%(self.epoch))

if __name__=="__main__":
    data_path = r'G:\mansen\deeplearning\pytorch\data'
    Train(data_path=data_path,epoch=10,lr=0.001).train()



