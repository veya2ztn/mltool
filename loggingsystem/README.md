# Loggingsystem
```
from mltool.loggingsystem import LoggingSystem
from mltool.dataaccelerate import infinite_batcher,DataSimfetcher
logsys            = LoggingSystem(True,"log/test")
metric_list       = ['loss']
metric_dict       = logsys.initial_metric_dict(metric_list)
master_bar        = logsys.create_master_bar(30)
#work only for jupyter notebook
master_bar.set_multiply_graph(figsize=(12,4),engine=[['plot']*len(metric_list)],labels=[metric_list])
for _ in master_bar:
    infiniter = DataSimfetcher(data_loader_train)
    inter_b    = logsys.create_progress_bar(len(data_loader_train))
    while inter_b.update_step():
        X_train, y_train = infiniter.next()
        X_train, y_train = X_train.to(device), y_train.to(device)
        loss,outputs     = model(X_train,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        _= logsys.metric_dict.update({'loss':loss.item()},inter_b.now)
        #work only for jupyter notebook
        master_bar.update_graph_multiply([[logsys.metric_dict.metric_dict['runtime']['loss']]])
```
