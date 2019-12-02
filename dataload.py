import numpy as np
import torch.utils.data as Data
import os
def get_axis_aligned_bbox(corner_4_coor):
    assert len(corner_4_coor)==8
    x1,y1,x2,y2,x3,y3,x4,y4=corner_4_coor
    x_coor=[x1,x2,x3,x4]
    y_coor=[y1,y2,y3,y4]
    left,right=min(x_coor),max(x_coor)
    top,down  =min(y_coor),max(y_coor)
    c_x,c_y=(left+right)/2,(top+down)/2
    w,h=abs(right-left),abs(down-top)
    return c_x,c_y,w,h
#test test
#from RPN_utils import get_axis_aligned_bbox
def get_image_box_pair(file_dir):
    boxed_file_name='groundtruth.txt'
    gt_path=os.path.join(file_dir,boxed_file_name)
    jpg_list=[name for name in os.listdir(file_dir) if '.jpg' in name]
    gt_boxes=[]
    with open(gt_path) as f:
        for line in f:
            gt_boxes.append([float(data) for data in line.split(',')])

    if len(gt_boxes)==len(jpg_list):
        image_good_pair=[0]*len(gt_boxes)
        for name in jpg_list:
            path= os.path.join(file_dir,name)
            #path= name
            num = int(name.strip('.jpg'))-1
            bbox= gt_boxes[num]
            image_good_pair[num]=[path,get_axis_aligned_bbox(bbox),bbox]
        return image_good_pair
    else:
        print('bad number match at {}'.format(file_dir))
        return []

class COCODataLoader:
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        self.image_files = [path.strip() for path in img_files]
        self.label_files = [
            path.replace("Images", "Labels").replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").strip()
            for path in self.image_files
        ]
        self.datasize = len(self.image_files)

    def get_sample(self):
        now_index=np.random.choice(self.datasize,1)[0]
        now_image=self.image_files[now_index]
        now_taget=self.label_files[now_index]
        return now_image,now_taget

class MyDataLoader(Data.Dataset):
    def __init__(self,dataset_path):
        classes=os.listdir(dataset_path)
        img_pair_for={}
        for _class in os.listdir(dataset_path):
            file_dir=os.path.join(dataset_path,_class)
            if not os.path.isfile(file_dir):
                img_pair_for[file_dir]=get_image_box_pair(file_dir)
        self.img_pair_for=img_pair_for
        self.classes=list(img_pair_for.keys())
        self.one_class   = False
        self.isfix_frame = False
        self.fixed_frame = 10
        self.max_jump    = 10
        self.imgprocesser= None
        self.length      = 10000

    def set_one_class(self,key):
        if key not in self.classes:
            print("This class is not in the dist")
            raise NotImplementedError
        self.the_class = key

    def get_sample(self):
        if self.one_class:
            now_class=self.the_class
        else:
            now_class=np.random.choice(self.classes,1)[0]
        img_pairs=self.img_pair_for[now_class]
        pairs_num=len(img_pairs)
        if self.isfix_frame:
            delta_templete_n = self.fixed_frame
        else:
            delta_templete_n = np.random.randint(self.max_jump)

        now_detection_num=np.random.randint(delta_templete_n,pairs_num)
        now_templete_num =now_detection_num-delta_templete_n
        now_detection    =img_pairs[now_detection_num]
        now_templete     =img_pairs[now_templete_num]
        return now_templete,now_detection

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        We generate train set every time, so the index here is meanless
        """
        if self.one_class:
            now_class=self.the_class
        else:
            now_class=np.random.choice(self.classes,1)[0]
        img_pairs=self.img_pair_for[now_class]
        pairs_num=len(img_pairs)
        if self.isfix_frame:
            delta_templete_n = self.fixed_frame
        else:
            delta_templete_n = np.random.randint(self.max_jump)

        now_detection_num=np.random.randint(delta_templete_n,pairs_num)
        now_templete_num =now_detection_num-delta_templete_n
        now_detection    =img_pairs[now_detection_num]
        now_templete     =img_pairs[now_templete_num]
        if self.imgprocesser is None:
            raise NotImplementedError
        img_path,target_region,shape=now_templete
        x,y = self.imgprocesser.template(img_path,target_region)
        img_path,target_should,shape=now_detection
        z,t = self.imgprocesser.detected(img_path,target_region,target_should)
        x = self.imgprocesser.torchdataform(x)
        y = self.imgprocesser.torchdataform(y)
        z = self.imgprocesser.torchdataform(z)
        t = self.imgprocesser.torchdataform(t)
        return (x,y,z),t
