import argparse
import yaml
import mindspore
from dataset import *
from model import STgramMFN, ArcMarginProduct
from trainer import *
import mindspore.dataset as ds
from mindspore.train.callback import LossMonitor
config_path = './config.yaml'
with open(config_path) as f:
    param = yaml.safe_load(f)

parser = argparse.ArgumentParser(description='STgram-MFN')

# data dir
parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model saved dir')
parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result saved dir')
parser.add_argument('--data-dir', default=param['data_dir'], type=str, help='data dir')
parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='processing data saved dir')
parser.add_argument('--process-machines', default=param['process_machines'], type=list,
                    help='allowed processing machines')
parser.add_argument('--ID-factor', default=param['ID_factor'], help='times factor for different machine types and ids to label')
# extract feature
parser.add_argument('--sr', default=param['sr'], type=int, help='sample rate of wav files')
parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT param: n_fft')
parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT param: n_mels')
parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT param: hop_length')
parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT param: win_length')
parser.add_argument('--power', default=param['power'], type=float, help='STFT param: power')
parser.add_argument('--frames', default=param['frames'], type=int, help='split frames')
parser.add_argument('--skip-frames', default=param['skip_frames'], type=int, help='skip frames in spliting')
# train
parser.add_argument('--num-class', default=param['num_class'], help='number of labels')
parser.add_argument('--epochs', default=param['epochs'], type=int)
parser.add_argument('--workers', default=param['workers'], type=int, help='number of workers for dataloader')
parser.add_argument('--batch-size', default=param['batch_size'], type=int)
parser.add_argument('--lr', '--learning-rate', default=param['lr'], type=float)
parser.add_argument('--log-every-n-steps', default=20, type=int)
parser.add_argument('--save-every-n-epochs', default=param['save_every_n_epochs'], type=int, help='save encoder and decoder model every n epochs')
parser.add_argument('--early-stop', default=param['early_stop'], type=int, help='number of epochs for early stopping')
parser.add_argument('--device-ids', default=param['device_ids'])
# arcface
parser.add_argument('--arcface', default=True, type=bool, help='using arcface or not')
parser.add_argument('--m', type=float, default=param['margin'], help='margin for arcface')
parser.add_argument('--s', type=float, default=param['scale'], help='scale for arcface')

parser.add_argument('--version', default='STgram_MFN_ArcFace(m=0.7,s=30)', type=str,
                    help='trail version')






def preprocess():
    args = parser.parse_args()
    root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
    if not os.path.exists(root_folder):
        utils.path_to_dict(process_machines=args.process_machines,
                           data_dir=args.data_dir,
                           root_folder=root_folder,
                           ID_factor=args.ID_factor)


def test(args):
    if args.arcface:
        arcface = ArcMarginProduct(128, args.num_class, m=args.m, s=args.s)
    else:
        arcface = None
    model = STgramMFN(num_class=args.num_class,
                      c_dim=args.n_mels,
                      win_len=args.win_length,
                      hop_len=args.hop_length,
                      arcface=arcface)
    if mindspore.cuda.is_available() and len(args.device_ids) > 0:
        args.device = mindspore.device(f'cuda:{args.device_ids[0]}')
        mindspore.ops.GradOperation.deterministic = True
        mindspore.ops.GradOperation.benchmark = True
    else:
        args.device = mindspore.device('cpu')
        args.gpu_index = -1

    with mindspore.cuda.device(args.device_ids[0]):
        model_path = os.path.join(args.model_dir, args.version, f'checkpoint_best.pth.tar')
        model.load_state_dict(mindspore.load_checkpoint(model_path)['clf_state_dict'])
        args.dp = False
      #  if len(args.device_ids) > 1:
      #      args.dp = True
       #     model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=args.ID_factor,
                                       classifier=model,
                                       arcface=arcface,
                                       optimizer=None,
                                       scheduler=None,
                                       args=args)
        trainer.test(save=True)


def train(args):
    if args.arcface:
        arcface = ArcMarginProduct(128, args.num_class, m=args.m, s=args.s)
    else:
        arcface = None
    model = STgramMFN(num_class=args.num_class,
                      c_dim=args.n_mels,
                      win_len=args.win_length,
                      hop_len=args.hop_length,
                      arcface=arcface)
    # if mindspore.cuda.is_available() and len(args.device_ids) > 0:
    #     args.device = mindspore.device(f'cuda:{args.device_ids[0]}')
    #     mindspore.ops.GradOperation.deterministic = True
    #     mindspore.ops.GradOperation.benchmark = True
    # else:
    #     args.device = mindspore.device('cpu')
    #     args.gpu_index = -1

    root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
    clf_dataset = WavMelClassifierDataset(root_folder, args.sr, param['ID_factor'])
    train_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
  #  print(type(train_clf_dataset))
    #train_clf_loader = torch.utils.data.DataLoader
   # train_clf_loader = mindspore.utils.data.DataLoader(
 #   dataset = ds.batch(batch_size=args.batch_size)
 #    sampler = ds.SequentialSampler()
 #    dataset = ds.NumpySlicesDataset(train_clf_dataset, sampler=sampler)
 #    class DatasetGenerator:
 #        def __init__(self,
 #                     batch_size=args.batch_size,
 #                     train_clf_dataset=train_clf_dataset,
 #                     #shuffle=False,
 #                     #num_parallel_workers=args.workers,
 #                     pin_memory=True,
 #                     drop_last=True):
 #            self.batch_size=batch_size
 #            #self.shuffle=shuffle
 #            #self.num_parallel_worker=num_parallel_workers
 #            self.pin_memory=pin_memory
 #            self.drop_last=drop_last
 #            self.train_clf_dataset=train_clf_dataset
 #        def __getitem__(self, item):
 #            pass
 #        def __len__(self):
 #            return len(self.train_clf_dataset)
    #train_clf_loader1=DatasetGenerator()
    train_clf_loader=ds.GeneratorDataset(source=train_clf_dataset,
                                         num_parallel_workers=args.workers,
                                         column_names=None,
                                         drop_remainder=True,
                                         schema=None,
                                         shuffle=False)
    # train_clf_loader =ds.GeneratorDataset(
    #     train_clf_dataset,
    #     #batch_size=args.batch_size,
    #     shuffle=False,
    #     num_parallel_workers=args.workers)
    #     #pin_memory=True,
    #    # drop_last=True)
   # print(type(train_clf_dataset))
    optimizer = mindspore.nn.Adam(model.parameters(), lr=args.lr)
    #
    scheduler =mindspore.nn.cosine_decay_lr(optimizer, T_max=len(train_clf_loader), eta_min=0, last_epoch=-1)
    # 此处进行引入方便调试
    from mindspore import context, Tensor
    from mindspore.communication import init, get_rank

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


    init("nccl")
    value = get_rank()

    # with mindspore.cuda.device(args.device_ids[0]):
    args.dp = False
    #if len(args.device_ids) > 1:
     #   args.dp = True
     #   model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                   id_fctor=args.ID_factor,
                                   classifier=model,
                                   arcface=arcface,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   args=args)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = LossMonitor()
    trainer.train(train_clf_loader)



def main(args):
    preprocess()
    train(args)
    test(args)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
