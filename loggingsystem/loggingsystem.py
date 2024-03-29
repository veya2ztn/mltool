from .MLlog import ModelSaver,AverageMeter,RecordLoss,Curve_data,IdentyMeter,LossStores,anomal_detect
from ..fastprogress import master_bar, progress_bar,isnotebook,TqdmToLogger
from ..tableprint.printer import summary_table_info
from tqdm.contrib.logging import logging_redirect_tqdm
from tensorboardX import SummaryWriter
import tensorboardX
import os, random ,torch,shutil
import numpy as np
import torch.backends.cudnn as cudnn
import logging,time
os.environ['WANDB_CONSOLE']='off'
ISNotbookQ=isnotebook()

import wandb
class RNGSeed:
    def __init__(self, seed):
        self.seed = seed
        self.set_random_seeds()

    def set_random_seeds(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
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
    global_step = console = bar_log =file_logger= tqdm_out =runtime_log =None
    wandb_logs={}
    wandb_prefix=''
    bar_log_path=runtime_log_path=info_log_path=None
    def __init__(self,global_do_log,ckpt_root,info_log_path=None,bar_log_path=None,gpu=0,
                      project_name="project",seed=None, use_wandb=False, flag="",
                      verbose=True,recorder_list = ['tensorboard'],Q_batch_loss_record=False,disable_progress_bar=False):
        self.global_do_log   = global_do_log
        
        self.diable_logbar   = not global_do_log
        self.disable_progress_bar= not global_do_log or disable_progress_bar
        self.ckpt_root       = ckpt_root
        #self.Q_recorder_type = 'tensorboard'
        self.Q_batch_loss_record = Q_batch_loss_record
        self.gpu_now         = gpu
        self.to_hub_info     = {}
        if seed == 'random':seed = random.randint(1, 100000)
        if isinstance(seed,int):self.set_rdn_seed(seed)
        self.seed = seed
        if global_do_log:
            if verbose:print(f"log at {ckpt_root}")
            self.info_log_path = os.path.join(self.ckpt_root, f'{flag}log.info') if info_log_path is None else info_log_path
            self.bar_log_path  = os.path.join(self.ckpt_root, f'{flag}bar.logging.info') if bar_log_path is None else bar_log_path
            self.runtime_log_path= os.path.join(self.ckpt_root, f'{flag}runtime.log') if bar_log_path is None else bar_log_path
            self.recorder_list=recorder_list
            if use_wandb and use_wandb =='wandb_runtime':
                self.recorder_list.append('wandb_runtime')
            if use_wandb and use_wandb =='wandb_on_success':
                self.recorder_list.append('wandb_on_success')

    def reinitialize_on_path(self, new_path):
        if self.recorder is not None:  self.recorder.close()
        if self.master_bar is not None:self.master_bar.close()
        if self.train_bar is not None: self.train_bar.close()
        if self.valid_bar is not None: self.valid_bar.close()
        self.ckpt_root = new_path
        self.create_recorder(['tensorboard'])
        self.file_log = self.create_logger("information_file_log",console=True,offline_path=os.path.join(self.ckpt_root, f'log.info'),console_level=logging.WARNING)
    
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

    @staticmethod
    def create_logger(name,console=False,offline_path=None,console_level=logging.DEBUG,logformat='%(asctime)s %(message)s'):
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):logger.handlers.clear()# Important!!
        logger.setLevel(logging.DEBUG)
        
        if offline_path:
            info_dir,info_file = os.path.split(offline_path)
            try:# in multi process will conflict
                if info_dir and not os.path.exists(info_dir):os.makedirs(info_dir)
            except:
                pass
            handler = logging.FileHandler(offline_path)
            handler.setLevel(level = logging.DEBUG)
            handler.setFormatter(logging.Formatter(logformat))
            logger.addHandler(handler)

        if console:
            console = logging.StreamHandler();
            console.setLevel(console_level)
            console.setFormatter(logging.Formatter(logformat))
            logger.addHandler(console)
        return logger

    def wandb_watch(self,*args,**kargs):
        if ('wandb_runtime' in self.recorder_list) or ('wandb_on_success' in self.recorder_list):
            wandb.watch(*args,**kargs)

    def info(self,string,show=True):
        if not self.global_do_log:return
        if self.file_logger is None:
            self.file_log = self.create_logger("information_file_log",console=True,offline_path=self.info_log_path,console_level=logging.WARNING)
        if show:
            self.file_log.warn(string)
        else:
            self.file_log.info(string)

    def record(self,name,value,epoch=None,epoch_flag = 'epoch', extra_epoch_dict = None):
        if not self.global_do_log:return
        if 'tensorboard' in self.recorder_list:
            self.recorder.add_scalar(name,value,epoch)
        if 'naive' in self.recorder_list:
            self.recorder.step([value])
            self.recorder.auto_save_loss()
        if ('wandb_runtime' in self.recorder_list):
            record_dict = {name:value,epoch_flag:epoch}
            if extra_epoch_dict is not None:
                for key,val in extra_epoch_dict.items():
                     record_dict[key] = val
            self.wandblog(record_dict)
        if ('wandb_on_success' in self.recorder_list):
            if not hasattr(self,'wandb_logs'):self.wandb_logs={}
            if epoch_flag not in self.wandb_logs:self.wandb_logs[epoch_flag]={}
            if epoch not in self.wandb_logs[epoch_flag]:self.wandb_logs[epoch_flag][epoch]={}
            if name not in self.wandb_logs[epoch_flag][epoch]:self.wandb_logs[epoch_flag][epoch][name]=[]
            self.wandb_logs[epoch_flag][epoch][name].append(value) # if same epoch call multiple times, we collect and do mean
    

    def add_figure(self,name,figure,epoch):
        if not self.global_do_log:return
        if 'tensorboard' in self.recorder_list:
            self.recorder.add_figure(name,figure,epoch)
    
    def add_table(self,name,table,epoch,columns):
        if not self.global_do_log:return
        if ('wandb_runtime' in self.recorder_list) or ('wandb_on_success' in self.recorder_list):
            self.wandblog({name: wandb.Table(data=table,columns = columns),'epoch':epoch})

    def create_master_bar(self,batches,banner_info=None,offline_bar=False):
        if self.bar_log is None and (not isnotebook()) and self.bar_log_path:
            self.info(f"the bar log will also save in {self.bar_log_path}",show=False)
            self.bar_log = self.create_logger('progress_bar',console=True,offline_path=self.bar_log_path if offline_bar else None,logformat='%(message)s')
            self.tqdm_out = TqdmToLogger(self.bar_log,level=logging.DEBUG)
        if self.runtime_log is None and (not isnotebook()) and self.runtime_log_path:
            self.info(f"the runtime loss in train/valid iter will save in {self.runtime_log_path}",show=False)
            self.runtime_log = self.create_logger('runtime_log',console=False,offline_path=self.runtime_log_path)
        if banner_info is not None and self.global_do_log:
            if banner_info == 1:banner_info = self.banner.demo()
            self.info(banner_info)
        logging_redirect_tqdm([self.bar_log])

        if  isinstance(batches,int):batches=range(batches)
        elif isinstance(batches,list):pass
        else: raise NotImplementedError
        self.master_bar = master_bar(batches,lwrite_log=self.runtime_log,file=self.tqdm_out,bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}", disable=self.diable_logbar)
        return self.master_bar

    def create_progress_bar(self,batches,force=False,disable=False,**kargs):
        if self.progress_bar is not None and not force:
            _=self.progress_bar.restart(total=batches)
        else:
            self.progress_bar = progress_bar(range(batches),
                lwrite_log=self.runtime_log,file=self.tqdm_out,
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                disable=self.disable_progress_bar,parent=self.master_bar,**kargs)
        return self.progress_bar

    def create_model_saver(self,path=None,accu_list=None,earlystop_config={},
                                anormal_d_config={},**kargs):
        if not self.global_do_log:return
        self.saver_path  = os.path.join(self.ckpt_root,'saver') if not path else path
        self.model_saver = ModelSaver(self.saver_path,accu_list,earlystop_config=earlystop_config  ,**kargs)
        self.anomal_detecter =anomal_detect(**anormal_d_config)

    def create_recorder(self,recorder_list=None, **kargs):
        if not self.global_do_log:return
        if recorder_list is None: recorder_list = self.recorder_list
        if ('wandb_runtime' in recorder_list) or ('wandb_on_success' in recorder_list):
            print("we will use wandb")
            project = kargs.get('project')
            group   = kargs.get('group')
            job_type= kargs.get('job_type')
            name    = kargs.get('name')

            wandb.init(config  = kargs.get('args'),
                project = project,
                #entity  = "szztn951357",
                group   = group,
                job_type= job_type,
                name    = name,
                settings= wandb.Settings(_disable_stats=True),
                id = kargs.get('wandb_id'),
                resume=kargs.get('wandb_resume')
                )
        else:
            print(f"wandb is off, the recorder list is  {recorder_list}, we pass wandb")
        if 'tensorboard' in recorder_list and self.train_recorder is None:
            self.train_recorder = self.valid_recorder = self.create_web_inference(self.ckpt_root,**kargs)
        if 'naive' in recorder_list and self.train_recorder is None:
            self.train_recorder = RecordLoss(loss_records=[AverageMeter()],mode = 'train',save_txt_step=1,log_file = self.ckpt_root)
            self.valid_recorder = RecordLoss(loss_records=[IdentyMeter(),IdentyMeter()], mode='test',save_txt_step=1,log_file = self.ckpt_root)
        return self.train_recorder

    def create_web_inference(self,log_dir,hparam_dict=None,metric_dict=None,**kargs):
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

    def save_latest_ckpt(self,checkpointer,epoch,train_loss,doearlystop=True,saveQ=True, optimizer=None,**kargs):
        if not self.global_do_log:return
        bad_condition_happen = self.anomal_detecter.step(train_loss)
        if saveQ or bad_condition_happen:
            assert checkpointer is not None
            if isinstance(checkpointer,dict):
                checkpointer = self.complete_the_ckpt(checkpointer,optimizer=optimizer)
            else:
                checkpointer = checkpointer.module if hasattr(checkpointer,'module') else checkpointer
            self.model_saver.save_latest_model(checkpointer,epoch,**kargs)
        return (bad_condition_happen and doearlystop)

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
        archive_flag        = time.strftime("%m_%d_%H_%M_%S") if not archive_flag else archive_flag
        routine_weight_path,epoch = self.model_saver.get_latest_model(soft_mode=True)
        best_weight_path    = self.model_saver.get_best_model(key_flag=key_flag)
        new_saver_path      = self.saver_path+f".{archive_flag}"
        new_routine_weight_path = os.path.join(new_saver_path,routine_weight_path) if routine_weight_path is not None else None
        new_best_weight_path    = os.path.join(new_saver_path,best_weight_path) if best_weight_path is not None else None
        old_saver_path          = self.saver_path
        old_routine_weight_path = os.path.join(old_saver_path,routine_weight_path) if routine_weight_path is not None else None
        old_best_weight_path    = os.path.join(old_saver_path,best_weight_path) if best_weight_path is not None else None

        if not os.path.exists(new_saver_path):
            os.system(f"cp -r {old_saver_path} {new_saver_path}")
        #os.rename(self.saver_path,new_saver_path)
        self.model_saver._initial()
        return (old_routine_weight_path,epoch),old_best_weight_path

    def runtime_log_table(self,table_string):
        if not self.global_do_log:return
        if self.bar_log is None:
            if not ISNotbookQ:
                self.master_bar.write_table(table_string,2,4)
        else:
            magic_char = "\033[F"
            self.bar_log.info(magic_char * (len(table_string)+2))
            for line in table_string:
                self.bar_log.info(line)
    
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
        if 'tensorboard' in self.recorder_list:
            self.recorder.export_scalars_to_json(os.path.join(self.ckpt_root,"all_scalars.json"))

    def wandblog(self,pool,step=None):
        if ('wandb_runtime' in self.recorder_list) or ('wandb_on_success' in self.recorder_list):
            if self.wandb_prefix != "":
                pool = dict([(f"{self.wandb_prefix}_{k}",v) for k,v in pool.items()])
            wandb.log(pool,step=step)

    def close(self):
        if not self.global_do_log:return
        if self.recorder is not None:  self.recorder.close()
        if self.master_bar is not None:self.master_bar.close()
        if self.train_bar is not None: self.train_bar.close()
        if self.valid_bar is not None: self.valid_bar.close()
        if ('wandb_on_success' in self.recorder_list):
            epoch_flags = list(self.wandb_logs.keys())
            self.info(f'''we will now do wandb record processing and do record, may very cost time.
            we have collect {len(epoch_flags)} level data:{epoch_flags} with {[len(self.wandb_logs[k]) for k in epoch_flags]} records
            ''')
            for epoch_flag, pools in self.wandb_logs.items():
                for epoch, named_value_pool in pools.items():
                    logpool = {epoch_flag:epoch}
                    for name,value_list in named_value_pool.items():
                        logpool[name] = np.mean(value_list)
                self.wandblog(logpool)
        if ('wandb_runtime' in self.recorder_list) or ('wandb_on_success' in self.recorder_list):
            wandb.finish()    

    def initial_metric_dict(self,accu_list):
        self.accu_list   =  accu_list
        self.metric_dict =  MetricDict(accu_list)
        return self.metric_dict

    def banner_initial(self,epoches,FULLNAME,training_loss_trace=['loss'],show_best_accu_types=None,print_once=True):
        if not self.global_do_log:
            return
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
        if not self.global_do_log:
            return
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
