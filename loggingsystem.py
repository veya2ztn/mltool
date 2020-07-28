from .MLlog import ModelSaver,AverageMeter,RecordLoss,Curve_data,IdentyMeter,LossStores
from .fastprogress import master_bar, progress_bar,isnotebook
from tensorboardX import SummaryWriter
import tensorboardX
import os
class LoggingSystem:
    '''
    How to use:
        logsys = LoggingSystem()
        logsys.checkpoint.
        logsys.webtracker.
        logsys.terminaler.
        logsys.webtracker.
    '''
    def __init__(self,global_do_log,ckpt_root,gpu=0):
        self.global_do_log = global_do_log
        self.diable_logbar = not global_do_log
        self.ckpt_root     = ckpt_root
        self.progress_bar  = None
        self.recorder      = None
        self.Q_recorder_type = 'tensorboard'
        self.Q_batch_loss_record = False
        self.master_bar = None
        self.gpu_now    = gpu

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
        if banner_info is not None and self.global_do_log:print(banner_info)
        self.master_bar = master_bar(range(batches), disable=self.diable_logbar)
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

    def close(self):
        if not self.global_do_log:return
        if self.Q_recorder_type == 'tensorboard':
            self.recorder.export_scalars_to_json(os.path.join(self.ckpt_root,"all_scalars.json"))
            self.recorder.close()
        self.master_bar.close()
        self.train_bar.close()
        self.valid_bar.close()
