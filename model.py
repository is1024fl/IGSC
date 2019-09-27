import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque
from torchvision.models import resnet101 as resnet101
from torchvision.models import resnet152 as resnet152

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class classifier(nn.Module):

    def __init__(self, backbone=None, pretrained=True, freeze=True, k=20, d=300):
        super(classifier, self).__init__()
        self.k = k
        self.d = d
        self.non_linear_param = 0 if self.k == 1 else self.k + 1
        self.output_num = self.non_linear_param + self.k * (self.d + 1)

        self.backbone = None
        if backbone == "resnet101":
            self.backbone = resnet101(pretrained=pretrained)

        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained=pretrained)
        else:
            assert False, "backbone error"

        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.transform = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, self.output_num))

    def forward(self, visual, semantics):
        visual_feature = self.features(visual)

        self.visual_feature = visual_feature.view(visual_feature.size(0), -1)
        self.visual_matrix = self.transform(self.visual_feature)

        semantics = semantics.t()
        semantics = semantics.expand(self.visual_matrix.shape[0], self.d, -1)

        if self.non_linear_param == 0:
            self.matrix = self.visual_matrix[..., :-1].view(-1, self.k, self.d)
            self.bias = self.visual_matrix[..., -1].view(-1, self.k)[..., None]
            semantic_transforms = torch.matmul(self.matrix, semantics) + self.bias

        else:
            self.visual_outer = self.visual_matrix[..., :self.non_linear_param]
            self.visual_inner = self.visual_matrix[..., self.non_linear_param:]

            self.visual_outer_matrix = self.visual_outer[..., :-1].view(-1, 1, self.k)
            self.visual_outer_bias = self.visual_outer[..., -1].view(-1, 1)[..., None]

            self.visual_inner_matrix = self.visual_inner[..., :-self.k].view(-1, self.k, self.d)
            self.visual_inner_bias = self.visual_inner[..., -self.k:].view(-1, self.k)[..., None]

            semantic_transforms = torch.matmul(self.visual_inner_matrix, semantics) + self.visual_inner_bias
            semantic_transforms = torch.tanh(semantic_transforms)
            semantic_transforms = torch.matmul(self.visual_outer_matrix, semantic_transforms) + self.visual_outer_bias

        self.semantic_transforms = semantic_transforms.transpose(1, 2).contiguous()
        output = torch.softmax(self.semantic_transforms, dim=1)

        return output


class transformer(nn.Module):

    def __init__(self, backbone=None, linear=None, pretrained=True, freeze=True, k=20, d=300):
        super(transformer, self).__init__()

        self.d = d
        self.backbone = None

        if backbone == "resnet101":
            self.backbone = resnet101(pretrained=pretrained)

        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained=pretrained)

        else:
            assert False, "backbone error"

        self.output_num = 0
        if linear:
            self.linear = True
            self.k = k
            self.output_num = self.k * self.d
        else:
            self.linear = False
            self.k1 = k
            self.k2 = int(k * d / (k + d))
            self.output_num = self.k1 * self.k2 + self.k2 * self.d

        assert 0 < self.output_num <= k * d, "Output num error, should range in 0 ~ k * d"

        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.transform = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, self.output_num))

    def forward(self, visual, semantics, state=None):

        semantics = semantics.t()
        semantics = semantics.expand(visual.shape[0], self.d, -1)
        
        visual_feature = self.features(visual)
        self.visual_feature = visual_feature.view(visual_feature.size(0), -1)
        self.visual_matrix = self.transform(self.visual_feature)

        if self.linear:
            self.visual_matrix = self.visual_matrix.view(-1, self.k, self.d)

            self.semantic_transforms = torch.matmul(self.visual_matrix, semantics)
            self.semantic_transforms = self.semantic_transforms.norm(p=2, dim=1)
            self.semantic_transforms = self.semantic_transforms[:, :, None]

        else:
            self.outer_matrixs = self.visual_matrix[:, :self.k1 * self.k2].view(-1, self.k1, self.k2)
            self.inner_matrixs = self.visual_matrix[:, self.k1 * self.k2:].view(-1, self.k2, self.d)

            self.semantic_transforms_1 = torch.matmul(self.inner_matrixs, semantics)
            self.semantic_transforms_tanh = torch.tanh(self.semantic_transforms_1)
            self.semantic_transforms = torch.matmul(self.outer_matrixs, self.semantic_transforms_tanh)

            self.semantic_transforms = self.semantic_transforms.norm(p=2, dim=1)
            self.semantic_transforms = self.semantic_transforms[:, :, None]

        if state == "train":
            output = self.semantic_transforms

        elif state == "test":
            self.semantic_transforms = 1 / (self.semantic_transforms + 1e-9)
            output = torch.softmax(self.semantic_transforms, dim=1)

        else:
            assert False, "train/ test state error"

        return output


def model_epoch(epoch, loss_name, model, type, data_loader, concepts, optimizer, writer, **kwargs):

    print(loss_name)

    state = None
    if loss_name.find('train') != -1:
        state = "train"
        model.train()
        torch.set_grad_enabled(True)

    elif loss_name.find('test') != -1 or loss_name.find('val') != -1:
        state = "test"
        model.eval()
        torch.set_grad_enabled(False)

    else:
        assert False, ("Mode Error")

    metrics = {
        'total': deque(),
        'correct': deque(),
        'total_g': deque(),
        'correct_g': deque()
    }

    for batch_i, batch_data in enumerate(data_loader, 1):

        # input
        batch_img = batch_data['image'].to(DEVICE)
        batch_label = batch_data['label'].to(DEVICE)

        # conventional ZSL result
        gts = batch_label[:, concepts[loss_name]['label']][..., None]

        # cal loss
        if state == 'train':

            outputs = model(Variable(batch_img), concepts[loss_name]['vector'], "train")
            optimizer.zero_grad()

            if type == "classifier":
                loss = F.binary_cross_entropy(outputs, gts)

            elif type == "transformer":
                pos_i = np.where((gts == 1).squeeze().data.cpu())
                neg_i = np.where((gts == 0).squeeze().data.cpu())

                pos_trans = outputs[pos_i].view(outputs.shape[0], -1)
                neg_trans = outputs[neg_i].view(outputs.shape[0], -1)

                loss = torch.sum(torch.stack(
                    [torch.sum(torch.stack([torch.exp(p - n) for n in ns for p in ps]))
                     for ps, ns in zip(pos_trans, neg_trans)]))
                loss = torch.log(loss + torch.FloatTensor([1.0]).to(DEVICE))

            else:
                assert False, "model type error"

            loss.backward()
            optimizer.step()

            tmp_loss = loss.item()
            writer.add_scalar(loss_name + '_loss', tmp_loss, batch_i + (epoch - 1) * len(data_loader))
            print('[%d, %6d] loss: %.4f' % (epoch, batch_i * data_loader.batch_size, tmp_loss))

        # ZSL predict
        outputs = model(Variable(batch_img), concepts[loss_name]['vector'], "test")
        maxs = torch.max(outputs, 1)[1][..., None]
        maxs_onehot = torch.zero_(outputs).scatter_(1, maxs, 1)
        metrics['total'].extend(np.array(gts.tolist()))
        metrics['correct'].extend(np.array(maxs_onehot.tolist()))

        # GZSL result
        gts_g = batch_label[:, concepts['general']['label']][..., None]
        outputs_g = model(Variable(batch_img), concepts['general']['vector'], "test")

        # calibration
        if 'calibration_gamma' in kwargs.keys():
            train_name = [k for k in concepts.keys() if k.find('train') >= 0][0]
            outputs_g[:, concepts[train_name]['label']] -= kwargs['calibration_gamma']

        # GZSL predict
        maxs_g = torch.max(outputs_g, 1)[1][..., None]
        maxs_g_onehot = torch.zero_(outputs_g).scatter_(1, maxs_g, 1)
        metrics['total_g'].extend(np.array(gts_g.tolist()))
        metrics['correct_g'].extend(np.array(maxs_g_onehot.tolist()))

        if 'debug' in kwargs.keys():
            break

    return metrics
