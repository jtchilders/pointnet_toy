from torch.utils import data as td
import torch


def get_dataset(config):

   ds = Dataset(config)

   sampler = None
   shuffle = config['data_handling']['shuffle']
   if config['hvd'] is not None:
      sampler = td.distributed.DistributedSampler(ds,num_replicas=config['nranks'],rank=config['rank'])
      shuffle = False

   loader = td.DataLoader(ds,
                          batch_size=config['training']['batch_size'],
                          shuffle=shuffle,
                          sampler=sampler,
                          num_workers=config['data_handling']['workers'])

   return loader


class Dataset(td.Dataset):
   def __init__(self,config):
      super(Dataset,self).__init__()

      self.dataset_size = config['data_handling']['dataset_size']
      self.input_shape = config['data_handling']['image_shape']
      self.num_classes = config['data_handling']['num_classes']

   def __len__(self):
      return self.dataset_size

   def __getitem__(self,i):
      inputs = torch.rand(self.input_shape)
      targets = torch.randint(0,self.num_classes,(self.input_shape[1],),dtype=torch.int32)
      weights_len = torch.randint(0,self.input_shape[1],(1,)).item()
      weights = torch.zeros((self.input_shape[1],))
      weights[:weights_len] = torch.ones((weights_len,))
      return inputs,weights,targets
