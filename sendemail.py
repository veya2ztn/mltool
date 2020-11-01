import smtplib
from email.mime.text import MIMEText
import json
import os

def read_json(path):
    with open(path,'r') as f:return json.load(f)
#设置服务器所需信息
#163邮箱服务器地址
mail_host = 'smtp.163.com'
#163用户名
mail_user = 'zhangtianning110'
#密码(部分邮箱为授权码)
mail_pass = 'CEACVHUFOEYPBYUJ'
#邮件发送方邮箱地址
sender = 'zhangtianning110@163.com'
#邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
receivers = ['645506775@qq.com']

#设置email信息
#邮件内容设置
CONStatusRecorder=".ConnectionStatus.json"
def send_message(text,mode):
    
    if os.path.exists(CONStatusRecorder):
        connection = read_json(CONStatusRecorder)["status"]
    else:
        connection = 1
    if connection < 1:return

    if mode in ['success','fail'] :
        message = MIMEText('project name:<| '+text+' |>','plain','utf-8')
        message['Subject'] = '[Trainning][Success]' if mode=='success' else '[Trainning][Fail]'
    elif mode == 'free':
        index = text  #it should be 0,1,2,3...
        content = f'Follow GPU node is  free in last 15 minites.\n {text} '
        message = MIMEText(text,'plain','utf-8')
        message['Subject'] = f'[Remind][FreeNow!][Athena:{text}]'
    elif mode == 'finish':
        index = text  #it should be 0,1,2,3...
        content = f'Follow GPU node is no activitiy in last 5 minites. Maybe its done or finish.\n {text} '
        message = MIMEText(text,'plain','utf-8')
        message['Subject'] = f'[Remind][Finish][Athena:{text}]'

    #发送方信息
    message['From'] = sender
    message['To'] = receivers[0]

    try:
        smtpObj = smtplib.SMTP()
        #连接到服务器
        smtpObj.connect(mail_host,25)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass)
        #发送
        smtpObj.sendmail(sender,receivers,message.as_string())
        smtpObj.quit()
        print('sand a {} message'.format("success" if mode=="success" else "fail"))
    except smtplib.SMTPException as e:
        connection={"status":0}
        with open(CONStatusRecorder,'w') as f:json.dump(connection,f)
        print('error',e) #打印错误

import sys
if __name__=="__main__":
    assert len(sys.argv)==3
    text = sys.argv[1]
    mode = sys.argv[2]
    send_message(text,mode)
