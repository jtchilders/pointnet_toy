import torch,logging
import numpy as np
logger = logging.getLogger(__name__)


num_classes = 0


def get_loss(config):
   global num_classes
   num_classes = config['data_handling']['num_classes']

   if 'loss' not in config:
      raise Exception('must include "loss" section in config file')

   if 'func' not in config['loss']:
      raise Exception('must include "func" loss section in config file')

   if config['loss']['func'] in globals():
      logger.info('getting loss function: %s',config['loss']['func'])
      return globals()[config['loss']['func']]
   else:
      raise Exception('%s loss function is not recognized; globals = %s' % (config['loss']['func'],globals()))


def get_accuracy(config):

   if 'loss' not in config:
      raise Exception('must include "loss" section in config file')

   if 'acc' not in config['loss']:
      raise Exception('must include "acc" loss section in config file')

   if config['loss']['acc'] in globals():
      logger.info('getting accuracy function: %s',config['loss']['acc'])
      return globals()[config['loss']['acc']]
   else:
      raise Exception('%s accuracy function is not recognized' % (config['loss']['acc'],globals()))


def pointnet_class_loss(pred,targets,end_points,reg_weight=0.001,device='cpu'):
   criterion = torch.nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
   classify_loss = criterion(pred, targets)
   
   # Enforce the transformation as orthogonal matrix
   mat_loss = 0

   if 'feature_trans' in end_points:
      tran = end_points['feature_trans']

      diff = torch.mean(torch.bmm(tran, tran.permute(0, 2, 1)), 0)
      mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1],device=device))

   # print('criterion = %s mat_loss = %s' % (classify_loss.item(),mat_loss.item()))
   loss = classify_loss + mat_loss * reg_weight

   return loss


def multiclass_acc(pred,targets):

   pred = torch.softmax(pred,dim=1)
   pred = pred.argmax(dim=1).float()

   eq = torch.eq(pred,targets.float())

   return torch.sum(eq).float() / float(targets.shape[0])


def pixel_wise_accuracy(pred,targets,device='cpu'):
   # need to calculate the accuracy over all points

   pred_stat = torch.nn.Softmax(dim=1)(pred)
   _,pred_value = pred_stat.max(dim=1)

   correct = (targets.long() == pred_value).sum()
   total = float(pred_value.numel())

   acc = correct.float() / total

   return acc


def mean_class_iou(pred,targets,device='cpu'):

   nclasses = pred.shape[1]
   npoints = targets.shape[1]
   nbatch = targets.shape[0]

   targets_onehot = torch.zeros(nbatch,nclasses,npoints,device=device,requires_grad=False)
   targets_onehot = targets_onehot.scatter_(1,targets.view(nbatch,1,npoints).long(),1).float()

   pred = torch.nn.functional.softmax(pred,dim=1)

   iou = IoU_coeff(pred,targets_onehot,device=device)

   return iou


def IoU_coeff(pred,targets,smooth=1,device='cpu'):
   intersection = torch.abs(targets * pred).sum(dim=2)
   union = targets.sum(dim=2) + pred.sum(dim=2) - intersection
   iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
   return iou


def dice_coef(pred,targets,smooth=1,device='cpu'):
   intersection = (targets * pred).sum(dim=2)
   union = targets.sum(dim=2) + pred.sum(dim=2)
   dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
   logger.info(' dice = %s ',dice)
   return dice.mean()


def pixelwise_crossentropy_focal(pred,targets,endpoints,weights,device='cpu',gamma=2.,alpha=1.):
   # from https://github.com/clcarwin/focal_loss_pytorch
   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # weights.shape = [N_batch,N_points]

   #logger.info('pred = %s  targets = %s weights = %s',pred.shape,targets.shape,weights.shape)

   nclasses = pred.shape[1]

   pred = pred.transpose(1,2)  # [N_batch,N_points,N_class]
   pred = pred.contiguous().view(-1,nclasses)  # [N_batch*N_points,N_class]

   weights = weights.view(-1)

   logpt = torch.nn.functional.log_softmax(pred,dim=1)  # [N_batch*N_points,N_class]
   logpt = logpt.gather(1,targets.view(-1,1).long())  # [N_batch*N_points,1]
   logpt = logpt.view(-1)
   pt = torch.autograd.Variable(logpt.data.exp())

   loss = -1 * (1 - pt) ** gamma * logpt * weights

   return loss.mean()


def pixelwise_crossentropy_weighted(pred,targets,endpoints,weights=None,device='cpu'):
   # for semantic segmentation, need to compare class
   # prediction for each point AND need to weight by the
   # number of pixels for each point

   # flatten targets and predictions

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]

   proportional_weights = []
   for i in range(num_classes):
      proportional_weights.append((targets == i).sum())
   proportional_weights = torch.Tensor(proportional_weights).to(device)
   proportional_weights = proportional_weights.sum() / proportional_weights
   proportional_weights[proportional_weights == float('Inf')] = 0

   loss_value = torch.nn.CrossEntropyLoss(weight=proportional_weights,reduction='none')(pred,targets.long())

   loss_value = loss_value * weights
   loss_value = torch.mean(loss_value)

   return loss_value
