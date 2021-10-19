from time import time
from sys import stdout
from warnings import warn
import shutil,os
import tqdm
import numpy as np
import io
#__all__ = ['master_bar', 'progress_bar', 'IN_NOTEBOOK']

CLEAR_OUTPUT = True
NO_BAR = False
#WRITER_FN = print
SAVE_PATH = None
SAVE_APPEND = False
MAX_COLS = 160
RegistSize ={'plot':(6,4),'imshow':(4,4)}
def isnotebook():
    try:
        from google import colab
        return True
    except: pass
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook, Spyder or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

IN_NOTEBOOK = isnotebook()
if IN_NOTEBOOK:
    try:
        from IPython.display import clear_output, display, HTML
        import matplotlib.pyplot as plt
    except:
        warn("Couldn't import ipywidgets properly, progress bar will use console behavior")
        IN_NOTEBOOK = False

def format_time(t):
    t = int(t)
    h,m,s = t//3600, (t//60)%60, t%60
    if h!= 0: return f'{h}:{m:02d}:{s:02d}'
    else:     return f'{m:02d}:{s:02d}'

class ProgressBar():
    update_every = 0.2

    def __init__(self, gen=None, total=None, display=True, leave=True, parent=None, auto_update=True):
        if gen is None:
            self._gen  = range(total)
            self.total = total
        else:
            self._gen  = gen
            self.total = len(gen)
        self.auto_update = auto_update
        self.parent = parent
        self.last_v = 0
        if parent is None: self.leave,self.display = leave,display
        else:
            self.leave,self.display=False,False
            parent.add_child(self)
        self.comment = ''
        if not self.auto_update:
            self.on_iter_begin()
            self.update(0)
        self.lwrite = print
        self.endQ   = False
        self.now    = 0

    def on_iter_begin(self): pass
    def on_interrupt(self): pass
    def on_iter_end(self): pass
    def on_update(self, val, text): pass

    def __iter__(self):
        self.on_iter_begin()
        self.update(0)
        try:
            for i,o in enumerate(self._gen):
                if i >= self.total: break
                yield o
                if self.auto_update: self.update(i+1)
        except:
            self.on_interrupt()
            raise
        self.on_iter_end()

    def update(self, val):
        if val == 0:
            self.start_t = self.last_t = time()
            self.pred_t,self.last_v,self.wait_for = 0,0,1
            self.update_bar(0)
        elif val >= self.last_v + self.wait_for or val == self.total:
            cur_t = time()
            avg_t = (cur_t - self.start_t) / val
            self.wait_for = max(int(self.update_every / (avg_t+1e-8)),1)
            self.pred_t = avg_t * self.total
            self.last_v,self.last_t = val,cur_t
            self.update_bar(val)
            if not self.auto_update and val >= self.total:
                self.on_iter_end()

    def update_bar(self, val):
        elapsed_t = self.last_t - self.start_t
        remaining_t = format_time(self.pred_t - elapsed_t)
        elapsed_t = format_time(elapsed_t)
        end = '' if len(self.comment) == 0 else f' {self.comment}'
        if self.total == 0:
            warn("Your generator is empty.")
            self.on_update(0, '100% [0/0]')
        else: self.on_update(val, f'{100 * val/self.total:.2f}% [{val}/{self.total} {elapsed_t}<{remaining_t}{end}]')

    def update_step(self,val=1):
        if self.now >= self.total:
            return False
        self.on_iter_begin()
        self.update(self.now)
        self.on_iter_end()
        self.now+=val
        return True

    def restart(self,gen=None,total=None):
        display= self.display
        leave=self.leave
        parent=self.parent
        auto_update=self.auto_update
        self.__init__(gen, total=total, display=display, leave=leave, parent=parent, auto_update=auto_update)
class MasterBar():
    def __init__(self, gen, cls, total=None):
        self.first_bar = cls(gen, total=total, display=False)
        self.lwrite = print
        self.now = 0
        self.total = total if total is not None else len(gen)

    def __iter__(self):
        self.on_iter_begin()
        for o in self.first_bar:
            self.now = o
            yield o
        self.on_iter_end()

    def on_iter_begin(self): self.start_t = time()
    def on_iter_end(self): pass
    def add_child(self, child): pass
    def write(self, line):      pass
    def update_graph(self, graphs, x_bounds, y_bounds): pass
    def update(self, val): self.first_bar.update(val)
    def update_step(self,val):
        self.on_iter_begin()
        self.update(val)
        self.on_iter_end()

def html_progress_bar(value, total, label, interrupted=False):
    bar_style = 'progress-bar-interrupted' if interrupted else ''
    return f"""
    <div>
        <style>
            /* Turns off some styling */
            progress {{
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }}
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {{
                background: #F44336;
            }}
        </style>
      <progress value='{value}' class='{bar_style}' max='{total}', style='width:300px; height:20px; vertical-align: middle;'></progress>
      {label}
    </div>
    """

def text2html_table(items):
    "Put the texts in `items` in an HTML table."
    html_code = f"""<table border="1" class="dataframe">\n"""
    html_code += f"""  <thead>\n    <tr style="text-align: left;">\n"""
    for i in items[0]: html_code += f"      <th>{i}</th>\n"
    html_code += f"    </tr>\n  </thead>\n  <tbody>\n"
    for line in items[1:]:
        html_code += "    <tr>\n"
        for i in line: html_code += f"      <td>{i}</td>\n"
        html_code += "    </tr>\n"
    html_code += "  </tbody>\n</table><p>"
    return html_code

class NBProgressBar(ProgressBar):
    def __init__(self, gen=None, total=None, display=True, leave=True, parent=None, auto_update=True,**kargs):
        self.progress = html_progress_bar(0, len(gen) if total is None else total, "")
        super().__init__(gen, total, display, leave, parent, auto_update)

    def on_iter_begin(self):
        if self.display: self.out = display(HTML(self.progress), display_id=True)
        self.is_active=True

    def on_interrupt(self):
        self.on_update(0, 'Interrupted', interrupted=True)
        if self.parent is not None: self.parent.on_interrupt()
        self.on_iter_end()

    def on_iter_end(self):
        if not self.leave and self.display:
            if CLEAR_OUTPUT: clear_output()
            else: self.out.update(HTML(''))
        self.is_active=False

    def on_update(self, val, text, interrupted=False):
        self.progress = html_progress_bar(val, self.total, text, interrupted)
        if self.display: self.out.update(HTML(self.progress))
        elif self.parent is not None: self.parent.show()

class NBMasterBar(MasterBar):
    names = ['train', 'valid']
    def __init__(self, gen=None, total=None, hide_graph=False, order=None, clean_on_interrupt=False, total_time=False,**kargs):
        super().__init__(gen, NBProgressBar, total)
        self.report,self.clean_on_interrupt,self.total_time = [],clean_on_interrupt,total_time
        self.text,self.lines = "",[]
        self.html_code = '\n'.join([self.first_bar.progress, self.text])
        if order is None: order = ['pb1', 'text', 'pb2']
        self.inner_dict = {'pb1':self.first_bar.progress, 'text':self.text}
        self.hide_graph,self.order = hide_graph,order
        self.multiply_graph_set=False

    def on_iter_begin(self):
        super().on_iter_begin()
        self.out = display(HTML(self.html_code), display_id=True)

    def on_interrupt(self):
        if self.clean_on_interrupt: self.out.update(HTML(''))

    def on_iter_end(self):
        if hasattr(self, 'imgs_fig'):
            plt.close()
            self.imgs_out.update(self.imgs_fig)
        if hasattr(self, 'graph_fig'):
            plt.close()
            self.graph_out.update(self.graph_fig)
        total_time = format_time(time() - self.start_t)
        if self.text.endswith('<p>'): self.text = self.text[:-3]
        if self.total_time: self.text = f'Total time: {total_time} <p>' + self.text
        self.out.update(HTML(self.text))

    def add_child(self, child):
        self.child = child
        self.inner_dict['pb2'] = self.child.progress
        if hasattr(self,'out'):self.show()

    def show(self):
        self.inner_dict['pb1'], self.inner_dict['text'] = self.first_bar.progress, self.text
        if 'pb2' in self.inner_dict: self.inner_dict['pb2'] = self.child.progress
        to_show = [name for name in self.order if name in self.inner_dict.keys()]
        self.html_code = '\n'.join([self.inner_dict[n] for n in to_show])
        if hasattr(self,"out"):self.out.update(HTML(self.html_code))

    def write(self, line, table=False):
        if not table: self.text += line + "<p>"
        else:
            self.lines.append(line)
            self.text = text2html_table(self.lines)

    def show_imgs(self, imgs, titles=None, cols=4, imgsize=4, figsize=None):
        if self.hide_graph: return
        rows = len(imgs)//cols if len(imgs)%cols == 0 else len(imgs)//cols + 1
        plt.close()
        if figsize is None: figsize = (imgsize*cols, imgsize*rows)
        self.imgs_fig, imgs_axs = plt.subplots(rows, cols, figsize=figsize)
        if titles is None: titles = [None] * len(imgs)
        for img, ax, title in zip(imgs, imgs_axs.flatten(), titles): img.show(ax=ax, title=title)
        for ax in imgs_axs.flatten()[len(imgs):]: ax.axis('off')
        if not hasattr(self, 'imgs_out'): self.imgs_out = display(self.imgs_fig, display_id=True)
        else: self.imgs_out.update(self.imgs_fig)

    def update_graph(self, graphs, x_bounds=None, y_bounds=None, figsize=(6,4)):
        if self.hide_graph: return
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        if len(self.names) < len(graphs): self.names += [''] * (len(graphs) - len(self.names))
        for g,n in zip(graphs,self.names): self.graph_ax.plot(*g, label=n)
        self.graph_ax.legend(loc='upper right')
        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure)

    def set_multiply_graph(self,nrows=1,ncols=2, engine=None, labels=None,figsize=(6,4)):
        # engine demo [[plot,imshow],[plot,imshow]]
        if engine is None:
            engine = [['plot']*ncols]*nrows
        else:
            nrows = len(engine)  # default
            ncols = max([len(row) for row in engine])
        figsize_row = max([np.sum([RegistSize[e][0] for e in row]) for row in engine])
        figsize_col = nrows*4
        figsize = (figsize_row,figsize_col)
        self.multiply_graph_set = True
        self.graph_fig,  graph_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if isinstance(graph_axes,np.ndarray):
            if len(graph_axes.shape) == 1:graph_axes=np.array([graph_axes])
        else:
            graph_axes=np.array([[graph_axes]])
        self.graph_axes = graph_axes
        self.engine     = engine
        self.labels     = labels
        self.graph_out = display(self.graph_axes[0][0].figure, display_id=True)
        self.imgs_out=self.graph_out

    def update_graph_multiply(self, data):
        if self.hide_graph: return
        if not self.multiply_graph_set:
            print('please set mltiply graph first')
            raise NotImplementedError
        if not hasattr(self, 'graph_out'):
            self.graph_out = display(self.graph_axes[0][0].figure, display_id=True)
            self.imgs_out=self.graph_out
        graph_axes=self.graph_axes
        if not isinstance(self.graph_axes,list):
            if isinstance(self.graph_axes,np.ndarray):graph_axes=[self.graph_axes]
            else:graph_axes=[[self.graph_axes]]
        for i,axes_row in enumerate(self.graph_axes):
            for j, ax in enumerate(axes_row):
                ax.clear()
                d  = data[i][j]
                e  = self.engine[i][j]

                if e == 'plot': ax.plot(range(len(d)),d)
                elif e == 'imshow':
                    ax.imshow(d,cmap='hot',vmin=0, vmax=1)
                    ax.set_xticks(())
                    ax.set_yticks(())
                elif e == 'multiplot':
                    for line in d:ax.plot(range(len(d)),d)
                else:
                    raise NotImplementedError
                if self.labels is not None:
                    t  = self.labels[i][j]
                    ax.set_title(t)
        self.graph_out.update(ax.figure)
magic_char = "\033[F"
import sys
def multi_print_line(content, num=2,offset=0):
    """
    这个版本的print会在开始print的时候判断是否清除上一次的内容。
    当num=0的时候不会清除，而是自然输出

    """
    # 回到初始输出位置
    # back to the origin pos
    lines = len(content)
    if num>0:
        print(magic_char * (lines-1+offset), end="")
        sys.stdout.flush()

    if isinstance(content, list):
        for line in content:
            print(line)
    else:
        raise TypeError("Excepting types: list, dict. Got: {}".format(type(content)))

class tqdmBar(tqdm.tqdm):
    def __init__(self,*args,parent=None,**kargs):
        super(tqdmBar,self).__init__(*args,**kargs)
        self.now=0
        self.parent = parent

    def lwrite(self,text,end=None):
        self.set_description(text)

    @classmethod
    def write_table(self, table, num=2,offset=0,end="\n", nolock=False):
        """
        Print a message via tqdm (without overlap with bars)
        """

        tqdm.tqdm.write(magic_char * (len(table)+1), end="")
        for line in table:tqdm.tqdm.write(line)

    def print(self,content,num,offset=0):
        if isinstance(content,list):
            lines = len(content)
            content="\n".join(content)
        else:
            lines = 1
        if num>0:
            tqdm.tqdm.write(magic_char * (lines+offset), end="")
            sys.stdout.flush()
        tqdm.tqdm.write(content)

    def restart(self,total=1):
        if not self.disable:
            self.reset(total=total)
        else:
            self.total =total
        self.now = 0

    def update_step(self,val=1):
        if self.now >= self.total:
            self.refresh()
            #self.close()
            return False
        self.update(val)
        self.now+=val
        return True

def print_and_maybe_save(line,end=None):
    if end is None:print(line)
    else:print(line,end=end)
    if SAVE_PATH is not None:
        attr = "a" if os.path.exists(SAVE_PATH) else "w"
        with open(SAVE_PATH, attr) as f: f.write(line + '\n')

def printing():
    return False if NO_BAR else (stdout.isatty() or IN_NOTEBOOK)

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf + magic_char
    def flush(self):
        self.logger.log(self.level, self.buf)

if IN_NOTEBOOK: master_bar, progress_bar = NBMasterBar, NBProgressBar
else:           master_bar, progress_bar = tqdmBar, tqdmBar
