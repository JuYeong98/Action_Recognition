#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"

def weights_init(m):  # 모델의 weight 를 초기화한다. 
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:  
        m.weight.data.normal_(0.0, 0.02) #weight 를 0.0 , 0.02로 준다?
        if m.bias is not None: # 절편값이 없다면 0으로 채움 
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)  #weight 를 0.0 , 0.02로 준다?
        if m.bias is not None: # 절편값이 없다면 0으로 채움
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  #weight 를 1.0 , 0.02로 준다?
        m.bias.data.fill_(0) # 절편을 0으로 초기화

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition  스켈레톤 베이스 행동인식을 위한 프로세서
    """
  
    def load_model(self):  # 모델을 불러옴
        self.model = self.io.load_model(self.arg.model,  
                                        **(self.arg.model_args))
        self.model.apply(weights_init)    # 모델에 초기화된 weight를 적용함
        self.loss = nn.CrossEntropyLoss()  # 손실값은 CrossEntropyLoss를 적용
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':  # 테스트에 적용된 옵티마이저는 SGD(default)임 
            self.optimizer = optim.SGD(  
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.90,  # 모멘텀 값을 0.92로 변경 default 는 0.9
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam': 
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                betas =(0.9 , 0.999),
                eps = 1e-08,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        print("rank:", rank) # 각 카테고리별 정답을 확인하기 위해 추가한 코드
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        print(hit_top_k) #각 레이블별 TRUE / FALSE 볼 수 있는 코드 
        self.io.print_log('\tTop{}: {:.4f}%'.format(k, 100 * accuracy))  # Top1~ Top5 까지의 정확도를 보여줌.

    def top_k_by_category(label, score, top_k):
        instance_num, class_num = score.shape
        rank = score.argsort()
        hit_top_k = [[] for i in range(class_num)]
        for i in range(instance_num):
            l = label[i]
            hit_top_k[l].append(l in rank[i, -top_k:])

        accuracy_list = []
        for hit_per_category in hit_top_k:
            if hit_per_category:
                accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
            else:
                accuracy_list.append(0.0)
        print(accuracy_list)        
        return accuracy_list

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []  # loss value를 담을 리스트 생성
        scaler = torch.cuda.amp.GradScaler()
        #scaler = torch.cuda.amp.GradScaler() #추가 Mixed Precision
        for data, label in loader:

            # get data
          
            #label = list(label)  accumulation 인듯
            #for i in range(len(label)):  accumulation 인듯
                #label[i] = float(label[i])  accumulation 인듯

            #label = torch.tensor(label)  # list 였던 label 이 pytorch의 tensor 자료형으로 변환됨
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # self.model = nn.DataParallel(self.model)
            # self.model.cuda()
            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            """
            accumulation_steps =2
            self.model.zero_grad()
            #model.zero_grad()                                   
            for i, (data, label) in enumerate(loader):
                data = data.cuda() #cpu 와 gpu가 동시에 사용되는 에러를 해결하기 위해 추가한 코드
                label = label.cuda()  #cpu 와 gpu가 동시에 사용되는 에러를 해결하기 위해 추가한 코드
                predictions = self.model(data)             #forward pass         
                loss = self.loss(predictions, label)      #loss 계산
                loss = loss / accumulation_steps          #loss를 accumulation_steps로 나눈다.   
                loss.backward()                  #backward pass               
                if (i+1) % accumulation_steps == 0:            #해당 처음으로 돌아가 반복함. 
                    self.optimizer.step()                            
                    self.model.zero_grad() 
            """
            """
            self.model.zero_grad()
            output = self.model(data) 
            loss = self.loss(output, label)   
            """
            
            
            
            
            
            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)  #평균 loss를 저장
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()  #모델 평가
        loader = self.data_loader['test'] 
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            
            # get data
            data = data.float().to(self.dev)
            
            label = list(label)
            for i in range(len(label)):
                label[i] = float(label[i])

            label = torch.tensor(label)
            #print("label : ", label)
            # print("type(label) : ", type(label))
            print("type(label[0]): {}, label[0]: {}".format(type(label[0]),label[0]))
            label = label.long().to(self.dev)

            # inference

            
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy().tolist())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()
            self.show_topk(1)
            #self.show_topk(2)
            #self.show_topk(3)
            #self.show_topk(4)
            
            # show top-k accuracy
            #for k in self.arg.show_topk:
                #self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
