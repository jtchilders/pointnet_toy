import torch
import inspect


def get_optimizer(config):
   optimizers = inspect.getmembers(torch.optim,inspect.isclass)
   names = [o[0] for o in optimizers]

   if config['optimizer']['name'] in names:
      return optimizers[names.index(config['optimizer']['name'])][1]
   else:
      raise Exception('%s is not defined in pytorch: %s' % (config['optimizer']['name'],names))
