import smtplib
from email.mime.text import MIMEText
#设置服务器所需信息
#163邮箱服务器地址
mail_host = 'smtp.163.com'
#163用户名
mail_user = 'zhangtianning110'
#密码(部分邮箱为授权码)
mail_pass = 'sz3035286'
#邮件发送方邮箱地址
sender = 'zhangtianning110@163.com'
#邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
receivers = ['645506775@qq.com']

#设置email信息
#邮件内容设置


def send_message(text,successQ):
    message = MIMEText('project name:<| '+text+' |>','plain','utf-8')
    #邮件主题
    message['Subject'] = '[Trainning][Success]' if successQ else '[Trainning][Fail]'
    #发送方信息
    message['From'] = sender
    #接受方信息
    message['To'] = receivers[0]
    try:
        smtpObj = smtplib.SMTP()
        #连接到服务器
        smtpObj.connect(mail_host,25)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass)
        #发送
        smtpObj.sendmail(
            sender,receivers,message.as_string())
        #退出
        smtpObj.quit()
        print('sand a {} message'.format("success" if successQ else "fail"))
    except smtplib.SMTPException as e:
        print('error',e) #打印错误

import sys
if __name__=="__main__":
    assert len(sys.argv)>2
    text    = sys.argv[1]
    successQ= sys.argv[2]
    send_message(text,successQ)
