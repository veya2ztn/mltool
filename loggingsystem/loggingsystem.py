from .MLlog import ModelSaver,AverageMeter,RecordLoss,Curve_data,IdentyMeter,LossStores
from ..fastprogress import master_bar, progress_bar,isnotebook,TqdmToLogger
from ..tableprint.printer import summary_table_info
from tqdm.contrib.logging import logging_redirect_tqdm
from tensorboardX import SummaryWriter
import tensorboardX
import os, random ,torch,shutil
import numpy as np
import torch.backends.cudnn as cudnn
import logging,time

class RNGSeed:
    def __init__(self, seed):
        self.seed = seed
        self.set_random_seeds()

    def set_random_seeds(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        torch.manual_seed(seed)
        cudnn.enabled = True
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    def state_dict(self):
        rng_states = {
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
        }
        return rng_states

    def load_state_dict(self, rng_states):
        random.setstate(rng_states["random_state"])
        np.random.set_state(rng_states["np_random_state"])
        torch.set_rng_state(rng_states["torch_random_state"])
        torch.cuda.set_rng_state_all(rng_states["torch_cuda_random_state"])

class MetricDict:
    def __init__(self,accu_list,show_best_accu_types=None):
        self.accu_list   = accu_list
        self.metric_dict = {}
        self.show_best_accu_types = show_best_accu_types if show_best_accu_types is not None else accu_list
        if not isinstance(self.show_best_accu_types,list):self.show_best_accu_types=[self.show_best_accu_types]
        self.recorderer  = {}
        self.initial()

    def initial(self):
        accu_list = self.accu_list
        metric_dict={'training_loss': None,"runtime":{}}

        for accu_type in accu_list:
            metric_dict[accu_type]                  = -1.0
            metric_dict['best_'+accu_type]          = dict([[key,np.inf] for key in self.accu_list])
            metric_dict['best_'+accu_type]['epoch'] = 0
        self.metric_dict = metric_dict

    def update(self,value_pool,epoch):
        update_accu = {}
        if epoch not in self.metric_dict["runtime"]:self.metric_dict["runtime"][epoch]={}
        for accu_type,val in value_pool.items():
            if accu_type not in self.metric_dict["runtime"]:
                self.metric_dict["runtime"][accu_type]={'epoch':[],'value':[]}
            self.metric_dict["runtime"][accu_type]['value'].append(val)
            self.metric_dict["runtime"][accu_type]['epoch'].append(epoch)
            if accu_type not in self.metric_dict:continue
            self.metric_dict[accu_type] = val
        # for accu_type in self.accu_list:
        #     self.metric_dict[accu_type]= value_pool[accu_type]
        for accu_type in self.accu_list:
            update_accu[accu_type]=False
            if self.metric_dict[accu_type]< self.metric_dict['best_'+accu_type][accu_type]:
                self.metric_dict['best_'+accu_type]['epoch']=epoch
                for accu_type_sub in self.accu_list:
                    self.metric_dict['best_'+accu_type][accu_type_sub] = self.metric_dict[accu_type_sub]
                update_accu[accu_type]=True
        return update_accu

    def state_dict(self):
        return self.metric_dict

    def load(self,state_dict):
        state_dict_a = self.metric_dict
        state_dict_b = state_dict
        same_structureQ= True
        for key,val in state_dict_b.items():
            if key in ["runtime"]:continue
            if key not in state_dict_a:
                self.info(f"unexcepted key:{key} for your design metric ")
                same_structureQ= False
            else:
                type1 = type(val)
                type2 = type(state_dict_a[key])
                if type1!=type2:
                    self.info(f"different type for val/key:{key} wanted{type2} but given{type1}")
                    same_structureQ= False
        for key,val in state_dict_a.items():
            if key in ["runtime"]:continue
            if key not in state_dict_b:
                self.info(f"wanted key:{key} for your design metric ")
                same_structureQ= False
        if not same_structureQ:
            self.info("detected disaccord state dict, abort load")
            raise
        else:
            self.metric_dict = state_dict

    def load_state_dict(self,state_dict):
        self.load(state_dict)

    @property
    def recorder_pool(self):
        return dict([[key,None]  for key in self.keys if 'best' not in key])
    @property
    def keys(self):
        return list(self.metric_dict.keys())

class LoggingSystem:
    '''
    import time
    import numpy as np
    from mltool.loggingsystem import LoggingSystem
    accu_list = ['a','b','c']

    logsys = LoggingSystem(True,"test")
    logsys.Q_batch_loss_record=True

    FULLNAME          = f"test-train_searched_model"

    metric_dict       = logsys.initial_metric_dict(accu_list)
    banner            = logsys.banner_initial(10,FULLNAME,show_best_accu_types=['a'])

    master_bar        = logsys.create_master_bar(10)
    logsys.train_bar  = logsys.create_progress_bar(1,unit=' img')
    logsys.valid_bar  = logsys.create_progress_bar(1,unit=' img')
    time.sleep(1)
    for epoch in master_bar:
        valid_acc_pool = dict([[key,np.random.rand()] for key in accu_list])
        update_accu    = logsys.metric_dict.update(valid_acc_pool,epoch)
        #if save_best_Q:train_utils.save(save_dir,epoch + 1,rng_seed,model,optimizer,metric_dict=metric_dict,level="best")
        logsys.banner_show(epoch,FULLNAME)
        time.sleep(1)
    '''
    accu_list = metric_dict = show_best_accu_types=rdn_seed = banner=None
    progress_bar =master_bar=train_bar=valid_bar   = None
    recorder = train_recorder = valid_recorder     = None
    global_step = console = bar_log = tqdm_out  =None

    def __init__(self,global_do_log,ckpt_root,info_log_path=None,bar_log_path=None,gpu=0,project_name="project",seed=None,verbose=True):
        self.global_do_log   = global_do_log
        self.diable_logbar   = not global_do_log
        self.ckpt_root       = ckpt_root
        self.Q_recorder_type = 'tensorboard'
        self.Q_batch_loss_record = False
        self.gpu_now         = gpu
        self.to_hub_info     = {}
        if seed == 'random':seed = random.randint(1, 100000)
        if isinstance(seed,int):self.set_rdn_seed(seed)
        self.seed = seed
        if verbose:print(f"log at {ckpt_root}")
        self.info_log_path = os.path.join(self.ckpt_root, 'log.info') if info_log_path is None else info_log_path
        self.bar_log_path = 'bar.logging.info' if bar_log_path is None else bar_log_path
    def set_rdn_seed(self,seed):
        self.rdn_seed = RNGSeed(seed)

    def train(self):
        if not self.global_do_log:return
        self.progress_bar = self.train_bar
        self.recorder     = self.train_recorder

    def eval(self):
        if not self.global_do_log:return
        self.progress_bar = self.valid_bar
        self.recorder     = self.valid_recorder

    def info(self,string,show=True):
        if self.console is None:
            info_dir,info_file = os.path.split(self.info_log_path)
            if info_dir and not os.path.exists(info_dir):os.makedirs(info_dir)
            logging.basicConfig(level    = logging.DEBUG,
                    format   = '%(asctime)s %(message)s',
                    datefmt  = '%m-%d %H:%M',
                    filename = self.info_log_path,
                    filemode = 'w');
            # define a Handler which writes INFO messages or higher to the sys.stderr

            console = logging.StreamHandler();
            console.setLevel(logging.WARN)
            console.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger('').addHandler(console)

            #self.console = console

        if show:
            logging.warn(string)
        else:
            logging.debug(string)


    def record(self,name,value,epoch):
        if not self.global_do_log:return
        if self.Q_recorder_type == 'tensorboard':
            self.recorder.add_scalar(name,value,epoch)
        else:
            self.recorder.step([value])
            self.recorder.auto_save_loss()

    def add_figure(self,name,figure,epoch):
        if not self.global_do_log:return
        if self.Q_recorder_type == 'tensorboard':
            self.recorder.add_figure(name,figure,epoch)

    def create_master_bar(self,batches,banner_info=None):
        if self.bar_log is None and (not isnotebook()):
            print(f"the bar log will also save in {self.bar_log_path}")
            info_dir,info_file = os.path.split(self.bar_log_path)
            if info_dir and not os.path.exists(info_dir):os.makedirs(info_dir)
            bar_log = logging.getLogger("progress_bar")
            bar_log.setLevel(logging.DEBUG)

            handler = logging.FileHandler(self.bar_log_path)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('%(message)s'))
            bar_log.addHandler(handler)

            console = logging.StreamHandler();
            console.setFormatter(logging.Formatter('%(message)s'))
            bar_log.addHandler(console)
            self.bar_log  = bar_log
            self.tqdm_out = TqdmToLogger(self.bar_log,level=logging.DEBUG)

        if banner_info is not None and self.global_do_log:
            if banner_info == 1:banner_info = self.banner.demo()
            self.info(banner_info)
        if   isinstance(batches,int):batches=range(batches)
        elif isinstance(batches,list):pass
        else:raise NotImplementedError
        logging_redirect_tqdm([self.bar_log])

        self.master_bar = master_bar(batches,file=self.tqdm_out,bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}", disable=self.diable_logbar)
        return self.master_bar

    def create_progress_bar(self,batches,force=False,**kargs):
        if self.progress_bar is not None and not force:
            _=self.progress_bar.restart(total=batches)
        else:
            self.progress_bar = progress_bar(range(batches),file=self.tqdm_out,bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                                             disable=self.diable_logbar,parent=self.master_bar,**kargs)
        return self.progress_bar

    def create_model_saver(self,path=None,accu_list=None,**kargs):
        if not self.global_do_log:return
        self.saver_path  = os.path.join(self.ckpt_root,'saver') if not path else path
        self.model_saver = ModelSaver(self.saver_path,accu_list,**kargs)

    def create_recorder(self,**kargs):
        if not self.global_do_log:return
        if self.Q_recorder_type == 'tensorboard':
            self.train_recorder = self.valid_recorder = self.create_web_inference(self.ckpt_root,**kargs)
            return self.train_recorder
        elif self.Q_recorder_type == 'naive':
            self.train_recorder = RecordLoss(loss_records=[AverageMeter()],mode = 'train',save_txt_step=1,log_file = self.ckpt_root)
            self.valid_recorder = RecordLoss(loss_records=[IdentyMeter(),IdentyMeter()], mode='test',save_txt_step=1,log_file = self.ckpt_root)
            return self.train_recorder

    def create_web_inference(self,log_dir,hparam_dict=None,metric_dict=None):
        if not self.global_do_log:return
        if hparam_dict is None:return
        self.regist({'hyparam':hparam_dict})
        exp, ssi, sei      = tensorboardX.summary.hparams(hparam_dict, metric_dict)
        self.recorder      = SummaryWriter(logdir= log_dir)
        self.recorder.file_writer.add_summary(exp)
        self.recorder.file_writer.add_summary(ssi)
        self.recorder.file_writer.add_summary(sei)
        return self.recorder

    def regist(self,_dict):
        for key,val in _dict.items():
            self.to_hub_info[key]=val

    def send_info_to_hub(self,hubfile):
        if 'score' not in self.to_hub_info:
            self.to_hub_info['score'] = self.metric_dict.metric_dict['best_'+self.accu_list[0]][self.accu_list[0]]
        timenow = time.asctime( time.localtime(time.time()))
        info_string = f"{timenow} {self.to_hub_info['task']} {self.to_hub_info['score']} {self.ckpt_root}\n"
        with open(hubfile,'a') as f:f.write(info_string)

    def complete_the_ckpt(self,checkpointer,optimizer=None):
        '''
        checkpointer = {
            "model": model,
            "metric_dict":metric_dict,
            "rdn_seed": self.rdn_seed, (option)
            "optimizer": optimizer,  (option)
            ...
        }
        '''
        assert isinstance(checkpointer,dict)
        #assert 'model' in checkpointer
        if self.rdn_seed is not None and ("rdn_seed" not in checkpointer):
            checkpointer["rdn_seed"] = self.rdn_seed
        if 'model' in checkpointer:
            if (optimizer is not None) or hasattr(model,'optimizer'):
                if 'optimizer' not in checkpointer:
                    checkpointer['optimizer']= model.optimizer if optimizer is None else optimizer
        if 'metric_dict' not in checkpointer:
            checkpointer['metric_dict']= self.metric_dict
        #print(checkpointer)
        return checkpointer

    def save_best_ckpt(self,model,accu_pool,epoch,**kargs):
        if not self.global_do_log:return False
        model = model.module if hasattr(model,'module') else model
        return self.model_saver.save_best_model(model,accu_pool,epoch,**kargs)

    def save_latest_ckpt(self,checkpointer,epoch,optimizer=None,**kargs):
        if not self.global_do_log:return
        assert checkpointer is not None
        if isinstance(checkpointer,dict):
            checkpointer = self.complete_the_ckpt(checkpointer,optimizer=optimizer)
        else:
            checkpointer = checkpointer.module if hasattr(checkpointer,'module') else checkpointer
        self.model_saver.save_latest_model(checkpointer,epoch,**kargs)

    def load_checkpoint(self,checkpointer,path):
        assert 'model' in checkpointer
        if not isinstance(checkpointer,dict):#checkpointer is model
            try:
                model.load_from(path)
                last_epoch = -1
            except:
                if 'state_dict' in state_dict:state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict)
                last_epoch = state_dict['epoch']
        else:
            checkpointer = self.complete_the_ckpt(checkpointer)
            state_dict = torch.load(path)
            self.info("need reload key:");self.info(checkpointer.keys())
            self.info("has key:");self.info(state_dict.keys())
            for key in checkpointer:
                self.info(f"reload {key}")
                checkpointer[key].load_state_dict(state_dict[key])
            last_epoch = state_dict['epoch']
        return last_epoch

    def archive_saver(self,archive_flag=None,key_flag=None):
        archive_flag     = time.strftime("%m_%d_%H_%M_%S") if not archive_flag else archive_flag
        routine_weight_path,epoch = self.model_saver.get_latest_model(soft_mode=True)
        best_weight_path    = self.model_saver.get_best_model(key_flag=key_flag)
        new_saver_path      = self.saver_path+f".{archive_flag}"
        routine_weight_path = os.path.join(new_saver_path,routine_weight_path) if routine_weight_path is not None else None
        best_weight_path    = os.path.join(new_saver_path,best_weight_path) if best_weight_path is not None else None
        os.rename(self.saver_path,new_saver_path)
        self.model_saver._initial()
        return (routine_weight_path,epoch),best_weight_path

    def runtime_log_table(self,table_string):
        if not self.global_do_log:return
        if self.bar_log is None:
            self.master_bar.write_table(table_string,2,4)
        else:
            magic_char = "\033[F"
            self.bar_log.info(magic_char * (len(table_string)+2))
            for line in table_string:self.bar_log.info(line)
    def batch_loss_record(self,loss_list):
        if not self.global_do_log:return
        if self.Q_batch_loss_record:
            if self.global_step is None:self.global_step=0
            for i,val in enumerate(loss_list):
                self.record(f'batch_loss_{i}',val,self.global_step)
            self.global_step+=1
        else:
            return

    def save_scalars(self):
        if not self.global_do_log:return
        if self.Q_recorder_type == 'tensorboard':
            self.recorder.export_scalars_to_json(os.path.join(self.ckpt_root,"all_scalars.json"))

    def close(self):
        if not self.global_do_log:return
        if self.recorder is not None:  self.recorder.close()
        if self.master_bar is not None:self.master_bar.close()
        if self.train_bar is not None: self.train_bar.close()
        if self.valid_bar is not None: self.valid_bar.close()

    def initial_metric_dict(self,accu_list):
        self.accu_list   =  accu_list
        self.metric_dict =  MetricDict(accu_list)
        return self.metric_dict

    def banner_initial(self,epoches,FULLNAME,training_loss_trace=['loss'],show_best_accu_types=None,print_once=True):
        self.info("initialize the log banner")
        self.show_best_accu_types = self.accu_list if show_best_accu_types is None else show_best_accu_types
        assert isinstance(self.show_best_accu_types,list)
        self.epoches= epoches
        header      = [f'epoches:{epoches}']+self.accu_list+training_loss_trace
        self.banner = summary_table_info(header,FULLNAME,rows=len(self.show_best_accu_types)+1)
        if print_once:print("\n".join(self.banner_str(0,FULLNAME)[0] ))

        return self.banner

    def banner_str(self,epoch,FULLNAME,train_losses=[-1]):
        assert self.banner is not None
        assert self.metric_dict is not None
        metric_dict = self.metric_dict.state_dict()
        accu_list   = self.accu_list
        shut_cut    = " ".join(["{:.4f}".format(s) for s in [metric_dict[key] for key in accu_list]+train_losses])
        shut_cut    = "epoch:{0:04d}/{0:04d} ".format(epoch+1,self.epoches) + shut_cut
        rows        = []
        show_row_run= [f"epoch:{epoch+1}"]+[metric_dict[key] for key in accu_list]+train_losses
        rows       += [show_row_run]

        for accu_type in self.show_best_accu_types:
            best_epoch = metric_dict['best_'+accu_type]['epoch']
            show_row_temp = [f'{accu_type}:{best_epoch}']+[metric_dict['best_'+accu_type][key] for key in accu_list]+[""]*len(train_losses)
            rows+=[show_row_temp]

        outstring =  self.banner.show(rows,title=FULLNAME)

        return outstring,shut_cut
    def banner_show(self,epoch,FULLNAME,train_losses=[-1]):
        outstring,shut_cut = self.banner_str(epoch,FULLNAME,train_losses)
        self.runtime_log_table(outstring)
        self.info(shut_cut,show=False)

if __name__ == "__main__":
    logsys            = LoggingSystem(True,'test')
    epoches = 20
    FULLNAME= "TEST"

    master_bar        = logsys.create_master_bar(epoches)
    metric_dict       = logsys.initial_metric_dict(['a','b','c'])
    metric_dict       = metric_dict.metric_dict
    _                 = logsys.create_recorder(hparam_dict={},metric_dict=metric_dict)
    logsys.train_bar  = logsys.create_progress_bar(1)
    logsys.valid_bar  = logsys.create_progress_bar(1)
    banner            = logsys.banner_initial(epoches,FULLNAME)
    for epoch in master_bar:
        logsys.train()
        inter_b    = logsys.create_progress_bar(20)
        while inter_b.update_step():time.sleep(0.2)
        logsys.eval()
        inter_b    = logsys.create_progress_bar(20)
        while inter_b.update_step():time.sleep(0.2)
        update_accu    = logsys.metric_dict.update({'a':np.random.random(),
                                                    'b':np.random.random(),
                                                    'c':np.random.random()},
                                                    epoch)
        logsys.banner_show(epoch,FULLNAME,train_losses=[np.random.random()])
