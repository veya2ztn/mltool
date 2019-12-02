import numpy as np
import torch
import cv2

def get_subwindow_tracking(image, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):

    if isinstance(pos, float):
        pos = [pos, pos]
    im=image.copy()
    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


class data_process:
    def __init__(self,template_size,detected_size):
        self.context_amount = 0.5
        self.avg_chans = 0
        self.template_size = template_size
        self.detected_size = detected_size
        self.debug = True
        self.hanning_for_template = True
        self.hanning_for_detected = True
        self.isbox = True
        self.distribution='normal'
        self.detected_offset_mode = None
    def numpy2torch(self,input_img):
        shape = input_img.shape
        input_img = input_img[None]
        if len(shape)==3:
            input_img = np.transpose(input_img, (0,3, 1, 2))
        else:
            input_img = input_img[None]
        input_img = torch.from_numpy(input_img).float()
        return input_img

    def torch2numpy(self,input_img):
        '''
        for (1,3,w,h)
        '''
        input_img = input_img[0]
        shape = input_img.shape
        if len(shape) == 3:
            input_img = input_img.permute(1,2,0)
        input_img = input_img.cpu().numpy()
        return input_img

    def torchdataform(self,input_img):
        shape = input_img.shape
        if len(shape)==3:
            input_img = np.transpose(input_img, (2, 0, 1))
        else:
            input_img = input_img[None]
        return input_img
    
    def convert_target(self,processed_target):
        '''
        Process for response field
        input : a mask or a image with target box (now we don't support 斜的 box)
        output: a density image, for center f(c_x,c_y) is 1 and 1/(10000\sigma) is the boundary
        '''

        isbox = self.isbox
        distribution = self.distribution
        #if box
        if isbox:
            c_x,c_y,c_w,c_h,processed_shape = processed_target
            g_x,g_y=processed_shape
            if distribution == 'normal':
                grid_x  = np.arange(g_x)
                grid_y  = np.arange(g_y)
                rs, cs  = np.meshgrid(grid_x, grid_y)
                sigmaw  = c_w/np.sqrt(2*np.log(10000))
                sigmah  = c_h/np.sqrt(2*np.log(10000))
                #y = 1/(2*np.pi*sigmaw*sigmah)*np.exp(-0.5*((1.0*(rs-c_x)/sigmaw)**2+(1.0*(cs-c_y)/sigmah)**2))
                y = np.exp(-0.5*((1.0*(rs-c_x)/sigmaw)**2+(1.0*(cs-c_y)/sigmah)**2))
            elif distribution == 'average':
                y = np.zeros(processed_shape)
                x1,x2 = int(c_x-c_w//2),int(c_x+c_w//2)
                y1,y2 = int(c_y-c_h//2),int(c_y+c_h//2)
                y[y1:y2,x1:x2]=1
            elif distribution == 'ratio_form':
                s_x = c_x / g_x
                s_y = c_y / g_y
                s_w = c_w / g_x
                s_h = c_h / g_y
                y   = np.array([s_x,s_y,s_w,s_h])
            elif distribution == 'feature_map':
                pass
            else:
                raise NotImplementedError
            return y
        else:
            return mask

    def template(self,im_path,target):
        '''
        process 1: crop with the around background
        process 2: padding 0
        process 3: cos window for smooth
        '''

        img = cv2.imread(im_path)
        c_x,c_y,c_w,c_h = target
        exemplar_size = self.template_size

        wc_z   = c_w + self.context_amount * (c_w+c_h)
        hc_z   = c_h + self.context_amount * (c_w+c_h)
        sc_z   = round(np.sqrt(wc_z * hc_z))
        scale_z= exemplar_size / sc_z
        z_crop = get_subwindow_tracking(img, (c_x,c_y) , exemplar_size,sc_z,self.avg_chans,out_mode='numpy')
        z_crop = z_crop/255
        win_patch = z_crop

        model_size= exemplar_size
        if self.hanning_for_template:
            cos_window = np.outer(np.hanning(model_size), np.hanning(model_size))
            win_patch  = np.multiply(z_crop, cos_window[:, :, None])


        processed_target= (model_size//2,model_size//2,c_w*scale_z,c_h*scale_z,(model_size,model_size))
        converted_target= self.convert_target(processed_target)

        ### record phase for debuging
        if self.debug:
            self.scale_z            = scale_z
            self.im_path_template   = im_path
            self.target_template    = target
            self.template_patch     = z_crop
            self.template_patch_win = win_patch
        return win_patch,converted_target

    def detected(self,im_path,search_region,target_should=None):
        '''
        process 1: crop with the around background
        process 2: padding 0
        process 3: cos window for smooth
        '''
        img = cv2.imread(im_path)
        c_x,c_y,c_w,c_h = search_region

        instance_size = self.detected_size
        exemplar_size = self.template_size

        wc_z   = c_w + self.context_amount * (c_w+c_h)
        hc_z   = c_h + self.context_amount * (c_w+c_h)
        sc_z   = round(np.sqrt(wc_z * hc_z))

        scale_z   = exemplar_size / sc_z
        d_search  = (instance_size - exemplar_size) / 2
        pad       = d_search / scale_z
        sc_x      = sc_z + 2 * pad
        if self.detected_offset_mode is None:
            offset_x=0
            offset_y=0
        elif self.detected_offset_mode is 'random':
            #random shift max 1/3 width
            offset_r=(np.random.rand(2)-0.5)/2
            offset_x=offset_r[0]*c_w
            offset_y=offset_r[1]*c_h
        else:
            raise NotImplementedError

        if target_should is not None:
            c_x_s,c_y_s,c_w_s,c_h_s = target_should
            c_x=(c_x_s+c_x)//2+offset_x
            c_y=(c_y_s+c_y)//2+offset_y
        else:
            c_x+=offset_x
            c_y+=offset_y
        x_crop    = get_subwindow_tracking(img, (c_x,c_y) , instance_size, round(sc_x),self.avg_chans,out_mode='numpy')
        z_crop    = x_crop/255
        win_patch = z_crop

        model_size = instance_size
        if self.hanning_for_detected:
            cos_window = np.outer(np.hanning(model_size), np.hanning(model_size))
            win_patch  = np.multiply(z_crop, cos_window[:, :, None])

        ### record phase for debuging
        if self.debug:
            self.scale_z            = scale_z
            self.im_path_detected   = im_path
            self.target_detected    = target_should
            self.region_detected    = search_region
            self.detected_patch     = z_crop
            self.detected_patch_win = win_patch

        if target_should is None:
            return win_patch

        c_x_s,c_y_s,c_w_s,c_h_s = target_should
        # the ground truth is define as the offset for last frame target
        # last frame target is shift to the center
        # but this relation can be broken, let the detected image can be any where
        t_x,t_y = model_size//2+(c_x_s-c_x)*scale_z,model_size//2+(c_y_s-c_y)*scale_z
        processed_target= (t_x,t_y,c_w_s*scale_z,c_h_s*scale_z,(model_size,model_size))
        converted_target= self.convert_target(processed_target)
        return win_patch,converted_target

class VOC_processer:
    def __init__(self,model_size):
        self.model_size = model_size

    def numpy2torch(self,input_img):
        shape = input_img.shape
        input_img = input_img[None]
        if len(shape)==3:
            input_img = np.transpose(input_img, (0,3, 1, 2))
        else:
            input_img = input_img[None]
        input_img = torch.from_numpy(input_img).float()
        return input_img

    def torch2numpy(self,input_img):
        '''
        for (1,3,w,h)
        '''
        input_img = input_img[0]
        shape = input_img.shape
        if len(shape) == 3:
            input_img = input_img.permute(1,2,0)
        input_img = input_img.cpu().numpy()
        return input_img

    def convert_target(self,processed_target):
        '''
        Process for response field
        input: a mask or a image with target box (now we don't support 斜的 box)
        output: a density image, for center f(c_x,c_y) is 1 and 1/(10000\sigma) is the boundary
        '''

        isbox        = self.isbox
        distribution = self.distribution
        #if box
        if isbox:
            c_x,c_y,c_w,c_h,processed_shape = processed_target
            g_x,g_y=processed_shape
            if distribution == 'normal':
                grid_x  = np.arange(g_x)
                grid_y  = np.arange(g_y)
                rs, cs  = np.meshgrid(grid_x, grid_y)
                sigmaw  = c_w/np.sqrt(2*np.log(10000))
                sigmah  = c_h/np.sqrt(2*np.log(10000))
                #y = 1/(2*np.pi*sigmaw*sigmah)*np.exp(-0.5*((1.0*(rs-c_x)/sigmaw)**2+(1.0*(cs-c_y)/sigmah)**2))
                y = np.exp(-0.5*((1.0*(rs-c_x)/sigmaw)**2+(1.0*(cs-c_y)/sigmah)**2))
            elif distribution == 'average':
                y = np.zeros(processed_shape)
                x1,x2 = int(c_x-c_w//2),int(c_x+c_w//2)
                y1,y2 = int(c_y-c_h//2),int(c_y+c_h//2)
                y[y1:y2,x1:x2]=1
            return y
        else:
            return mask

    def convert_data(self,im_path,target):
        '''
        process 1: crop with the around background
        process 2: padding 0
        process 3: cos window for smooth
        '''

        img       = cv2.imread(im_path)
        h, w, _   = img.shape
        dim_diff  = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = cv2.resize(input_img, (self.model_size, self.model_size))
        # Channels-first
        input_img = input_img[None]
        input_img = np.transpose(input_img, (0,3, 1, 2))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        ### record phase for debuging
        if self.debug:
            self.scale_z            = scale_z
            self.im_path_template   = im_path
            self.target_template    = target
            self.template_patch     = z_crop
            self.template_patch_win = win_patch
        return win_patch,converted_target

    def detected(self,im_path,target):


        ### record phase for debuging
        if self.debug:
            self.scale_z            = scale_z
            self.im_path_detected   = im_path
            self.target_detected    = target_should
            self.region_detected    = search_region
            self.detected_patch     = z_crop
            self.detected_patch_win = win_patch

        if target_should is None:
            return win_patch

        c_x_s,c_y_s,c_w_s,c_h_s = target_should
        t_x,t_y = model_size//2+(c_x_s-c_x)*scale_z,model_size//2+(c_y_s-c_y)*scale_z
        processed_target= (t_x,t_y,c_w_s*scale_z,c_h_s*scale_z,(model_size,model_size))
        converted_target= self.convert_target(processed_target)
        return win_patch,converted_target


def get_batch_input(data_loader,img_processer,batch,cuda):
    xes = []
    yes = []
    zes = []
    tes = []
    for i in range(batch):
        templete_pair,detected_pair=data_loader.get_sample()

        img_path,target_region,shape=templete_pair
        x,y = img_processer.template(img_path,target_region)
        img_path,target_should,shape=detected_pair
        z,t = img_processer.detected(img_path,target_region,target_should)

        x,y = img_processer.numpy2torch(x),img_processer.numpy2torch(y)
        z,t = img_processer.numpy2torch(z),img_processer.numpy2torch(t)
        xes.append(x)
        yes.append(y)
        zes.append(z)
        tes.append(t)

    #return templates,detections,pos_neg_diffs
    if not cuda:
        return torch.cat(xes),torch.cat(yes),torch.cat(zes),torch.cat(tes)
    return torch.cat(xes).cuda(),torch.cat(yes).cuda(),torch.cat(zes).cuda(),torch.cat(tes).cuda()
