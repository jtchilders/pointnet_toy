#!/usr/bin/env python3
import argparse,logging,socket,json,sys,psutil
import numpy as np
import data_gen,pointnet,loss_acc,optimizer
logger = logging.getLogger(__name__)
import torch

def print_mem_cpu():
   start = time.time()
   while True:
      mem = psutil.virtual_memory()
      print(str(time.time()-start) + ' pid = ' + str(os.getpid()) + ' total mem = ' + str(mem.total) + ' free mem = ' + str(mem.free/mem.total*100.) + ' cpu usage = ' + str(psutil.cpu_percent()))
      time.sleep(1)

def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',help='configuration file in json format',required=True)

   parser.add_argument('--horovod',default=False, action='store_true', help="Setup for distributed training")

   parser.add_argument('--random_seed',default=0,type=int,help='numpy random seed')

   parser.add_argument('--mem_mon',default=False, action='store_true', help="spawn subprocess to monitor memory")

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   rank = 0
   nranks = 1
   hvd = None
   if args.horovod:
      import horovod.torch as hvd
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      logging_format = '%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s'

   if rank > 0 and logging_level == logging.INFO:
      logging_level = logging.WARNING
   
   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   np.random.seed(args.random_seed)

   config = json.load(open(args.config))
   
   # detect device available
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   config['rank'] = rank
   config['nranks'] = nranks
   config['hvd'] = hvd
   config['device'] = device

   logger.info('rank %s of %s',rank,nranks)
   logger.info('hostname:           %s',socket.gethostname())
   logger.info('python version:     %s',sys.version)
   logger.info('config file:        %s',args.config)
   logger.info('random_seed:        %s',args.random_seed)
   logger.info('horovod:            %s',args.horovod)
   logger.info('device:             %s',device)


   model = pointnet.PointNet1d_SemSeg(config).to(device)
   logger.info('got model')
   loss_func = loss_acc.get_loss(config)
   acc_func = loss_acc.get_accuracy(config)

   opt_class = optimizer.get_optimizer(config)
   opt = opt_class(model.parameters(),**config['optimizer']['args'])

   ds = data_gen.get_dataset(config)

   if rank == 0 and args.mem_mon:
      memorymon = mp.Process(target=print_mem_cpu)
      memorymon.start()

   accuracies = []
   losses = []
   epochs = config['training']['epochs']
   for epoch in range(epochs):
      logger.info('starting epoch %s of %s',epoch,config['training']['epochs'])

      for batch_number,(inputs,weights,targets) in enumerate(ds):

         inputs = inputs.to(device)
         weights = weights.to(device)
         targets = targets.to(device)

         logger.info('inputs = %s  weights = %s  targets = %s',inputs.shape,weights.shape,targets.shape)

         opt.zero_grad()

         pred,endpoints = model(inputs)
         loss = loss_func(pred,targets,endpoints,weights,device=device)

         loss.backward()
         opt.step()

         acc = acc_func(pred,targets,device)
         if 'mean_class_iou' in config['loss']['acc']:
            acc = acc.mean()
         accuracies.append(acc)
         losses.append(loss)
         if batch_number % config['training']['status'] == 0:
            acc = torch.median(torch.Tensor(accuracies))
            loss = torch.median(torch.Tensor(losses))
            logger.info('<[%3d of %3d, %5d of %5d]> train loss: %6.4f train acc: %6.4f ',
                        epoch + 1,epochs,batch_number,len(ds),loss.item(),acc.item())

            accuracies = []
            losses = []


if __name__ == "__main__":
   main()
