import os

def send_message(text,**kargs):
    out=os.system(f"curl http://149.129.63.78/tianning.send?text={text}")

import sys
if __name__=="__main__":
    assert len(sys.argv)==2
    text = sys.argv[1]
    send_message(text)
