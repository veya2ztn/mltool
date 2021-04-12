from .MLlog import ModelSaver,AverageMeter,RecordLoss,Curve_data,IdentyMeter,LossStores
from .fastprogress import master_bar, progress_bar,isnotebook
from .tableprint.printer import summary_table_info
from tensorboardX import SummaryWriter
import tensorboardX
import os
import numpy as np





class MetricDict:
    def __init__(self,accu_list,show_best_accu_types=None):
        self.accu_list   = accu_list
        self.metric_dict = {}
        self.show_best_accu_types = show_best_accu_types if show_best_accu_types is not None else accu_list
        if not isinstance(self.show_best_accu_types,list):self.show_best_accu_types=[self.show_best_accu_types]

        self.initial()

    def initial(self):
        accu_list = self.accu_list
        metric_dict={'training_loss': None}
        for accu_type in accu_list:
            metric_dict[accu_type]                  = -1.0
            metric_dict['best_'+accu_type]          = dict([[key,np.inf] for key in self.accu_list])
            metric_dict['best_'+accu_type]['epoch'] = 0
        self.metric_dict = metric_dict

    def update(self,value_pool,epoch):
        update_accu={}
        for accu_type in self.accu_list:self.metric_dict[accu_type]= value_pool[accu_type]
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
            if key not in state_dict_a:
                print(f"unexcepted key:{key} for your design metric ")
                same_structureQ= False
            else:
                type1 = type(val)
                type2 = type(state_dict_a[key])
                if type1!=type2:
                    print(f"different type for val/key:{key} wanted{type2} but given{type1}")
                    same_structureQ= False
        for key,val in state_dict_a.items():
            if key not in state_dict_b:
                print(f"wanted key:{key} for your design metric ")
                same_structureQ= False
        if not same_structureQ:
            print("detected disaccord state dict, abort load")
            raise
        else:
            self.metric_dict = state_dict


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
    accu_list = metric_dict = show_best_accu_types = None
    progress_bar =master_bar=train_bar=valid_bar   = None
    recorder = train_recorder = valid_recorder     = None
    global_step = None
    def __init__(self,global_do_log,ckpt_root,gpu=0,project_name="project",verbose=True):
        self.global_do_log = global_do_log
        self.diable_logbar = not global_do_log
        self.ckpt_root     = ckpt_root
        self.Q_recorder_type = 'tensorboard'
        self.Q_batch_loss_record = False
        self.gpu_now    = gpu
        if verbose:print(f"log at {ckpt_root}")

    def train(self):
        if not self.global_do_log:return
        self.progress_bar = self.train_bar
        self.recorder     = self.train_recorder

    def eval(self):
        if not self.global_do_log:return
        self.progress_bar = self.valid_bar
        self.recorder     = self.valid_recorder

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
        if banner_info is not None and self.global_do_log:
            if banner_info == 1:banner_info = self.banner.demo()
            print(banner_info)
        if   isinstance(batches,int):batches=range(batches)
        elif isinstance(batches,list):pass
        else:raise NotImplementedError
        self.master_bar = master_bar(batches, disable=self.diable_logbar)
        return self.master_bar

    def create_progress_bar(self,batches,force=False,**kargs):
        if self.progress_bar is not None and not force:
            _=self.progress_bar.restart(total=batches)
        else:
            self.progress_bar = progress_bar(range(batches), disable=self.diable_logbar,parent=self.master_bar,**kargs)
        return self.progress_bar

    def create_model_saver(self,path=None,accu_list=None,**kargs):
        if not self.global_do_log:return
        if path is None:path=self.ckpt_root
        self.model_saver = ModelSaver(path,accu_list,**kargs)

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
        exp, ssi, sei      = tensorboardX.summary.hparams(hparam_dict, metric_dict)
        self.recorder      = SummaryWriter(logdir= log_dir)
        self.recorder.file_writer.add_summary(exp)
        self.recorder.file_writer.add_summary(ssi)
        self.recorder.file_writer.add_summary(sei)
        return self.recorder

    def save_best_ckpt(self,model,accu_pool,epoch,**kargs):
        if not self.global_do_log:return False
        model = model.module if hasattr(model,'module') else model
        return self.model_saver.save_best_model(model,accu_pool,epoch,**kargs)

    def save_latest_ckpt(self,model,epoch,**kargs):
        if not self.global_do_log:return
        if model is None:raise
        model = model.module if hasattr(model,'module') else model
        self.model_saver.save_latest_model(model,epoch,**kargs)

    def runtime_log_table(self,table_string):
        if not self.global_do_log:return
        self.master_bar.write_table(table_string,2,4)

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
        print("initialize the log banner")
        self.show_best_accu_types = self.accu_list if show_best_accu_types is None else show_best_accu_types
        assert isinstance(self.show_best_accu_types,list)
        header      = [f'epoches:{epoches}']+self.accu_list+training_loss_trace
        self.banner = summary_table_info(header,FULLNAME,rows=len(self.show_best_accu_types)+1)
        if print_once:print("\n".join(self.banner_str(0,FULLNAME)))
        return self.banner

    def banner_str(self,epoch,FULLNAME,train_losses=[-1]):
        assert self.banner is not None
        assert self.metric_dict is not None
        metric_dict = self.metric_dict.state_dict()
        accu_list   = self.accu_list
        rows        = []
        show_row_run= [f"epoch:{epoch+1}"]+[metric_dict[key] for key in accu_list]+train_losses
        rows       += [show_row_run]

        for accu_type in self.show_best_accu_types:
            best_epoch = metric_dict['best_'+accu_type]['epoch']
            show_row_temp = [f'{accu_type}:{best_epoch}']+[metric_dict['best_'+accu_type][key] for key in accu_list]+[""]
            rows+=[show_row_temp]

        outstring =  self.banner.show(rows,title=FULLNAME)
        return outstring
    def banner_show(self,epoch,FULLNAME,train_losses=[-1]):
        outstring = self.banner_str(epoch,FULLNAME,train_losses)
        self.runtime_log_table(outstring)
