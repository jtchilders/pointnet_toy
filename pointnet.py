import torch
import logging
import building_blocks as bb
logger = logging.getLogger(__name__)


class PointNet1d(torch.nn.Module):
   def __init__(self,config):
      super(PointNet1d,self).__init__()

      input_shape = config['data_handling']['image_shape']

      assert(len(input_shape) == 2)

      nPoints = input_shape[1]
      nCoords = input_shape[0]
      nClasses = config['data_handling']['num_classes']
      
      logger.info('nPoints = %s, nCoords = %s, nClasses = %s',nPoints,nCoords,nClasses)

      self.input_trans = Transform1d(nPoints,nCoords,bn=False)

      self.input_to_feature = torch.nn.Sequential()
      for x in config['model']['input_to_feature']:
         N_in,N_out,bn,pool = x
         self.input_to_feature.add_module('conv_%d_to_%d' % (N_in,N_out),bb.Conv1d(N_in,N_out,bn=bn,pool=pool))

      self.feature_trans = Transform1d(nPoints,config['model']['input_to_feature'][-1][1],bn=False)

      self.feature_to_pool = torch.nn.Sequential()
      for x in config['model']['feature_to_pool']:
         N_in,N_out,bn,pool = x
         self.feature_to_pool.add_module('conv_%d_to_%d' % (N_in,N_out),bb.Conv1d(N_in,N_out,bn=bn,pool=pool))

      self.pool = torch.nn.MaxPool1d(nPoints)

      self.dense_layers = torch.nn.Sequential()
      for x in config['model']['dense_layers']:
         N_in,N_out,dropout,bn,act = x
         dr = int(dropout * 3)
         self.dense_layers.add_module('dense_%d_to_%d' % (N_in,N_out),bb.Linear(N_in,N_out,bn=bn,activation=act))
         if dropout > 0:
            self.dense_layers.add_module('dropout_%03d' % dr,torch.nn.Dropout(dropout))

      logger.info('model built')
      
   def forward(self,x):
      batch_size = x.shape[0]
      
      it = self.input_trans(x)

      x = torch.bmm(it,x)
      endpoints = {'input_trans':x}
      
      x = self.input_to_feature(x)
      
      ft = self.feature_trans(x)

      x = torch.bmm(ft,x)
      endpoints['feature_trans'] = x
      
      x = self.feature_to_pool(x)
      
      x = self.pool(x)
      
      x = x.reshape([batch_size,-1])
      endpoints['global_features'] = x
      
      x = self.dense_layers(x)
      
      return x,endpoints


class Transform1d(torch.nn.Module):
   """ Input (XYZ) Transform Net, input is BxNxK gray image
        Return:
            Transformation matrix of size KxK """
   def __init__(self,height,width,bn=False):
      super(Transform1d,self).__init__()

      self.width = width

      self.conv64 = bb.Conv1d(width,64,bn=bn,pool=False)
      self.conv128 = bb.Conv1d(64,128,bn=bn,pool=False)
      self.conv1024 = bb.Conv1d(128,1024,bn=bn,pool=False)

      self.pool = torch.nn.MaxPool1d(height)

      self.linear512 = bb.Linear(1024,512,bn=bn)
      self.linear256 = bb.Linear(512,256,bn=bn)

      self.linearK = torch.nn.Linear(256,width * width)
      self.linearK.bias = torch.nn.Parameter(torch.eye(width).view(width * width))

   def forward(self,x):
      batch_size = x.shape[0]
      
      x = self.conv64(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = x.reshape([batch_size,-1])
      x = self.linear512(x)
      x = self.linear256(x)
      
      x = self.linearK(x)
      x = x.reshape([batch_size,self.width,self.width])
      
      return x


class PointNet1d_SemSeg(torch.nn.Module):

   def __init__(self,config):
      super(PointNet1d_SemSeg,self).__init__()

      self.pointnet1d = PointNet1d(config)

      nClasses = config['data_handling']['num_classes']
      width = 64 + 1024

      self.conv512 = bb.Conv1d(width,512,bn=False,pool=False)
      self.conv256 = bb.Conv1d(512,256,bn=False,pool=False)
      self.conv128 = bb.Conv1d(256,128,bn=False,pool=False)
      self.convclass = bb.Conv1d(128,nClasses,bn=False,pool=False)

   def forward(self,x):

      image_classes,endpoints = self.pointnet1d(x)

      pointwise_features = endpoints['feature_trans']
      global_features = endpoints['global_features']
      global_features = global_features.view(global_features.shape[0],global_features.shape[1],1)
      global_features = global_features.repeat(1,1,pointwise_features.shape[-1])

      combined = torch.cat((pointwise_features,global_features),1)

      x = self.conv512(combined)
      x = self.conv256(x)
      x = self.conv128(x)
      x = self.convclass(x)

      return x,endpoints
