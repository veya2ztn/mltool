import torch
import torch.nn as nn
import numpy as np

import copy
import torch.nn.functional as F
import os

#######################################
#######  LOG_RECORDER   ###############
#######################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.win100=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #self.avg = self.sum / self.count
        self.win100.append(val)
        if len(self.win100) > 100:_=self.win100.pop(0)
        sum100=sum(self.win100)
        self.avg=1.0*sum100/len(self.win100)
        self.output = self.avg


class IdentyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val, n=1):
        self.val    = val
        self.output = self.val

class Curve_data(object):
    def __init__(self,max_leng = np.inf,ReduceQ=False,reducer=100,reduce_limit=10000):
        self.x = []
        self.y = []
        self.ml= max_leng
        if max_leng is np.inf:
            self.bound_x = None
            self.bound_y = None
            Reduce=False
        else:
            self.bound_x = [0,max_leng]
            self.bound_y = None
        self.reduceQ = ReduceQ
        self.reducer= reducer
        self.reduce_limit = reduce_limit

    @property
    def data(self):
        return [self.x,self.y]

    def reset(self):
        '''
        determinate the range of y via previous data
        '''
        y_win    = max(self.y)-min(self.y)
        now_at   = self.y[-1]
        self.y_bounds = [now_at-y_win, now_at+0.2*y_win]
        self.x = []
        self.y = []

    def reduce(self):
        if not self.reduce:return
        selector = np.linspace(0,self.reduce_limit-1,num=self.reducer).astype('int')
        self.x=np.array(self.x)
        self.y=np.array(self.y)
        self.y=self.y[selector].tolist()
        self.x=self.x[selector].tolist()


    def add_xy(self,xy):
        if len(self.x) > self.ml:self.reset()
        if len(self.x) > self.reduce_limit and self.reduceQ:self.reduce()
        self.x.append(xy[0])
        self.y.append(xy[1])

    def add_y(self,y):
        x = 0 if len(self.x)==0 else (self.x[-1]+1)%(self.ml+1)
        self.add_xy([x,y])

from .fastprogress import isnotebook

import tensorwatch as tw

IS_NOTEBOOK=isnotebook()

import collections

class RecordLoss:

    def __init__(self,loss_records=None,
                      graph_set=None,
                      mode = 'train',
                      save_txt_step=100,
                      save_model_step=2000,
                      log_file = 'log'):
        self.loss_records=loss_records
        self.graph_set   = graph_set
        self.initialQ    = False
        self.global_mode = False
        self.step_now    = 0
        self.save_txt_step = save_txt_step
        self.save_model_step = save_model_step
        self.auto_plot  = True
        self.mb = None
        self.method="" if not IS_NOTEBOOK else "notebook"
        ### force self.method="tensorwatch" if you want to use tensorwatch
        self.mode = mode
        self.LogFile = log_file
        if not os.path.exists(log_file):os.makedirs(log_file)
        self.model_loss_store=None
        self.watch = None
        self.streams= None
        self.twwatchQ=True
        self.auto_loss_saveQ = True
    def initial(self,num):
        if self.loss_records is None:self.loss_records = [AverageMeter() for i in range(num)]
        assert len(self.loss_records) == num
        if self.graph_set is None:self.graph_set = [Curve_data() for i in range(num)]
        assert len(self.graph_set) == num
        self.initialQ  = True

    def check_watch(self):
        if self.method == "tensorwatch" and self.watch is None:
            if self.mode == 'train':
                filenames = os.path.join(self.LogFile,'tw.train.log')
                if os.path.exists(filenames):os.remove(filenames)
                self.watch = tw.Watcher(filename=os.path.join(self.LogFile,'tw.train.log'),port=0)
            else:
                filenames = os.path.join(self.LogFile,'tw.valid.log')
                if os.path.exists(filenames):os.remove(filenames)
                self.watch = tw.Watcher(filename=os.path.join(self.LogFile,'tw.valid.log'),port=1)
            self.streams=[self.watch.create_stream(name='loss_'+str(i)) for i in range(len(self.loss_records))]

    def observe(self,**krage):
        if not self.twwatchQ:return
        if self.method == "tensorwatch":
            self.check_watch()
            self.watch.observe(**krage)
        else:
            if self.step_now ==1:
                print("in notebook mode, the watch will not open")
                print("if you want to test, let self.method = \'tensorwatch\'")
            return

    def stream_close(self):
        if self.watch is None:return
        if not self.watch.closed:
            if self.watch._clisrv is not None:
                self.watch._clisrv._stream.close()
                self.watch._clisrv.close()
            if self.watch._zmq_stream_pub is not None:
                self.watch._zmq_stream_pub.close()
            if self.watch._file is not None:
                self.watch._file.close()

    def record_latest_loss(self,loss_recorder):
        return loss_recorder.output

    def update_record(self,recorder,loss):
        if isinstance(loss,torch.Tensor):loss = loss.item()
        recorder.update(loss)

    def update(self,loss_list):
        data_list=[]
        if not self.initialQ:
            num_loss = len(loss_list)
            self.initial(num_loss)

        for loss_recorder,g,loss in zip(self.loss_records,self.graph_set,loss_list):
            self.update_record(loss_recorder,loss)# record data in recorder
            loss = self.record_latest_loss(loss_recorder)# record data until get enough for next statistic result
            if loss:g.add_y(loss)
            data_list.append(loss)

        if self.auto_plot and IS_NOTEBOOK and (self.mb is not None):self.update_graph(self.mb)

        if self.method == "tensorwatch" and self.twwatchQ:
            self.check_watch()
            for i,(s,data) in enumerate(zip(self.streams,data_list)):
                s.write((self.step_now,data))
            #self.watch.observe(losslist=data_list)
            #self.stream.write(data_list)

    def step(self,loss_list):
        self.update(loss_list)
        self.step_now +=1

    def update_graph(self,mb):
        graphs  =[[g.data]  for g in self.graph_set]
        x_bounds=[g.bound_x for g in self.graph_set]
        y_bounds=[g.bound_y for g in self.graph_set]
        mb.update_graph_multiply(graphs,x_bounds,y_bounds)
        #mb.update_graph(graphs, x_bounds, y_bounds)
            #return ll

    def auto_save_loss(self,path=None,file_name=None):
        if path is None:path = self.LogFile
        if not os.path.exists(path):os.makedirs(path)
        if file_name is None:file_name = self.mode+'.loss.log'
        file_name = os.path.join(path,file_name)
        if self.step_now%self.save_txt_step !=0: return
        # if self.step_now == 0:
        #     with open(file_name,'w') as log_file:
        #         log_file.write('')
        if self.auto_loss_saveQ:self.print2file(self.step_now,file_name)

    def auto_save_model(self,model,path,filename=None):
        if self.step_now%self.save_model_step !=0: return
        self.save_model(model,path,filename)

    def save_model(self,model,path,filename=None):
        if filename is None:filename = 'weights-{:07d}'.format(self.step_now)
        if not os.path.exists(path):os.makedirs(path)
        file_path = os.path.join(path,filename)
        #torch.save(model.state_dict(),file_path)
        model.save_to(file_path)
    def print2file(self,step,file_name):
        with open(file_name,'a') as log_file:
            ll=["{:.4f}".format(self.record_latest_loss(recorder)) for recorder in self.loss_records]
            printedstring=str(step)+' '+' '.join(ll)+'\n'
            #print(printedstring)
            _=log_file.write(printedstring)

    def save_best_model(self,model,score,num=3,path=None,filename=None,epoch=None,ForceQ=False):
        if self.mode is 'train' and not ForceQ:
            print('this mode is only design for valid, if you still want to use it, let ForceQ=True')
            raise
        if self.model_loss_store is None:self.model_loss_store=LossStores()
        if path is None: path = 'checkpoints'
        if not os.path.exists(path):os.makedirs(path)
        if epoch is None: epoch = self.step_now
        if filename is None: filename='epoch-{:04d}|l-{:.6f}'.format(epoch,score)
        stores = self.model_loss_store
        if len(stores.store)<num:
            file_path = os.path.join(path,filename)
            #torch.save(model.state_dict(),file_path)
            model.save_to(file_path)
            stores.update(score,filename)
            return
        if score < stores.min(num):
            file_path = os.path.join(path,filename)
            #torch.save(model.state_dict(),file_path)
            model.save_to(file_path)
            stores.update(score,filename)
        name_should_save = set(stores.minpart(num))
        name__now___save = set(os.listdir(path))
        name_should_del  = name__now___save.difference(name_should_save)
        for name in name_should_del:
            real_path = os.path.join(path,name)
            os.remove(real_path)

class LossStores:
    def __init__(self):
        self.store = collections.OrderedDict()
        self.last  = None
        self.nbm_c = 0
    def update(self,score,key_name):
        self.store[key_name] = score

    def reload(self):
        self.store = collections.OrderedDict()
    def minpart(self,num):
        sort=sorted(self.store.items(), key=lambda d: d[1])
        return [key for key,val in sort[:num]]

    def min(self,num):
        sort=sorted(self.store.items(), key=lambda d: d[1])
        return sort[num-1][1]

    def earlystop(self,num,max_length=20,anti_over_fit_length=30,mode="no_min_more"):
        self.buffer = list(self.store.values())
        if len(self.buffer)<=max_length:return False
        window = self.buffer[-max_length:]
        if mode == "no_min_more":
            #if num > max(window)*0.99:return True
            if num > max(window)-0.00001:return True
        anti_over_fit_min = self.buffer[-anti_over_fit_length:]
        if min(anti_over_fit_min)>min(self.buffer):return True
        return False

import time
class ModelSaver:
    '''
    Load the path
    '''
    record_file_name = "record.txt"
    def __init__(self,path,keep_latest_num=None, keep_best_num=None,
                          save_latest_form=None,save_best_form=None,
                          get_num=None,
                          es_max_window=20,es_mode="no_min_more"):

        self.root_path   = path
        self.latest_path = os.path.join(self.root_path,'routine')
        self.best_path   = os.path.join(self.root_path,'best')
        self.latest_record_file = os.path.join(self.latest_path,self.record_file_name)
        self.best_record_file   = os.path.join(self.best_path  ,self.record_file_name)
        self.early_stop_window  = es_max_window
        self.early_stop_mode    = es_mode
        if save_latest_form is None:save_latest_form = lambda x,y:"epoch-{}".format(x)
        if save_best_form   is None:save_best_form   = lambda x,y:"epoch-{}".format(x)
        if get_num is None:get_num = lambda x:x.split('|')[0].split('-')[1]

        self.save_best_form    = save_best_form
        self.save_latest_form  = save_latest_form
        self.get_num_f         = get_num
        self.keep_latest_num   = keep_latest_num
        self.keep_best_num     = keep_best_num
        self.model_loss_store  = LossStores()

    def _renewpath(self,path):
        self.root_path   = path
        self.latest_path = os.path.join(self.root_path,'routine')
        self.best_path   = os.path.join(self.root_path,'best')
        self.latest_record_file = os.path.join(self.latest_path,self.record_file_name)
        self.best_record_file   = os.path.join(self.best_path  ,self.record_file_name)

    def _checkpath(self):
        path = self.root_path
        project_name = ".".join(os.path.basename(path).split('.')[:-1])
        UPDIR=os.path.abspath(os.path.join(path, "../"))
        UPFILES = [f for f in os.listdir(UPDIR) if project_name in f]
        if len(UPFILES) != 0:
            print("detected same project checkpoints file")
            for f in UPFILES:
                print("  -{}".format(f))
            print("please manual set self._renewpath(newpath) ")
        if not os.path.exists(path):
            print("Path:{} are not exist, which mean this project does not run".format(self.root_path))
            raise NotImplementedError

    def _load(self,load_mode):
        if load_mode == 'latest':
            path  = self.latest_path
            record_file= self.latest_record_file
        elif load_mode == 'best':
            path  = self.best_path
            record_file= self.best_record_file
        else:
            raise NotImplementedError
        if not os.path.exists(path):os.makedirs(path)
        if not os.path.exists(record_file):
            with open(record_file, 'w') as f:
                f.write('#this is the record file for weight save\n')
                f.write('epoch-start\n')
        with open(record_file, 'r') as f:
            _list = [line.strip() for line in f]
            last_line = _list[-1]
            body_lines=_list[:-1]
            last_num  = self.get_num_f(last_line)
            self.body = '\n'.join(body_lines)
            self.model_last_num = int(last_num) if last_num != 'start' else -1
            self.weight_last_name = last_line if last_num != 'start' else None
            if load_mode == 'best':
                self.model_loss_store.reload()
                for line in body_lines:
                    if "#" in line : continue
                    line_list = line.strip().split(' ')
                    name = line_list[2]
                    score = float(line_list[3])
                    self.model_loss_store.update(score,name)
        return path,record_file

    def _reset(self,mode='backup',ForceQ=False):
        if mode == 'backup':
            print("initial backup mode, please clear manually if needed")
            self._backup()
        if mode == 'clear_all' and not ForceQ:
            print("Warnning: You are trying del weight file")
            print("Please use self._clear() manually")

    def _backup(self):
        time_now = str(int(time.time()))
        os.renames(path,path.strip('/')+'.backup.'+time_now)
        self._load()

    def _clear(self):
        os.removedirs(path)
        self._load()

    def _t2str(self,_input):
        if isinstance(_input,int):return "{:3d}".format(_input)
        if isinstance(_input,float):return "{:.6f}".format(_input)
        if isinstance(_input,str):return _input
        try:
            return self._t2str(float(_input))
        except:
            print("bad loss type",type(_input))
            raise

    def latest_weight_path(self,testQ=False):
        if testQ:self._checkpath()
        path,record_file=self._load('latest')
        last_weight_path=None
        if self.weight_last_name is not None:
            last_weight_path = os.path.join(path,self.weight_last_name)
        return self.model_last_num,last_weight_path

    def record_step(self,record_file,filename,other_info=""):
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = time_now +' '+filename+' '+ other_info
        last_line = filename
        with open(record_file, 'w') as f:
            f.write(self.body+'\n')
            f.write(text+'\n')
            f.write(last_line+'\n')

    def keep_latest(self,keep_num=None):
        if keep_num is None:keep_num = self.keep_latest_num
        if keep_num is None:return
        path = self.latest_path
        weight_name_list = os.listdir(path)
        weight_name_list = [name for name in weight_name_list if name != self.record_file_name]
        if len(weight_name_list)<=keep_num:return
        weight_n_name_list  = [(name,int(self.get_num_f(name))) for name in weight_name_list]
        weight_n_name_list.sort(key=lambda x:x[1])
        weight_del_list = weight_n_name_list[:-keep_num]
        for name,num in weight_del_list:
            real_path = os.path.join(path,name)
            os.remove(real_path)

    def save_latest_model(self,model,model_num_save=None,keep_num=None,other_info=""):
        path,record_file=self._load('latest')
        path = self.latest_path
        if model_num_save is None:
            model_num_now = self.model_last_num+1
            model_num_save= self.save_latest_form(model_num_now,other_info)
        real_path = os.path.join(path,model_num_save)
        #torch.save(model.state_dict(),real_path)
        model.save_to(real_path)
        self.record_step(record_file,model_num_save,other_info)
        self.keep_latest(keep_num)

    def best_weight_path(self,testQ=False):
        if testQ:self._checkpath()
        path,record_file=self._load('best')
        weight_best_name=self.model_loss_store.minpart(1)[0]
        weight_path=os.path.join(path,weight_best_name)
        return weight_path

    def save_best_model(self,model,score,keep_num=None,model_num_save=None,epoch=None,ForceQ=False,doearlystop=True):
        path,record_file=self._load('best')
        stores = self.model_loss_store
        num    = keep_num if keep_num is not None else self.keep_best_num
        if num is None:return

        if model_num_save is None:
            model_num_now = self.model_last_num+1
            model_num_save= self.save_best_form(model_num_now,score)


        if isinstance(score,list) or isinstance(score,np.ndarray):
            other_info = " ".join([self._t2str(s) for s in score])
            self.record_step(record_file,model_num_save,other_info)
            score = score[0]
        else:
            other_info = self._t2str(score)
            self.record_step(record_file,model_num_save,other_info)

        earlystopQ = stores.earlystop(score,max_length=self.early_stop_window,mode=self.early_stop_mode)
        if earlystopQ and doearlystop:return True

        stores.update(score,model_num_save)

        if model_num_now < num:
            file_path = os.path.join(path,model_num_save)
            #torch.save(model.state_dict(),file_path)
            model.save_to(file_path)
            return False

        if score < stores.min(num):
            file_path = os.path.join(path,model_num_save)
            #torch.save(model.state_dict(),file_path)
            model.save_to(file_path)

        name_should_save = set(stores.minpart(num)+[self.record_file_name])
        name__now___save = set(os.listdir(path))
        name_should_del  = name__now___save.difference(name_should_save)

        for name in name_should_del:
            real_path = os.path.join(path,name)
            os.remove(real_path)
        return False
