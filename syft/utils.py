from IPython.display import display
from IPython.html.widgets import IntProgress, HBox, HTML

import signal
import logging

class Progress(object):
    
    def __init__(self,start,end):

        self.pbar = IntProgress(orientation='horizontal',min=start, max=end)
        self.ptext = HTML()
        # Only way to place text to the right of the bar is to use a container
        container = HBox(children=[self.pbar, self.ptext])
        display(container)
    
    def update(self,value,stats):
        self.pbar.value = value
        out = "<div style='margin-left:5px; margin-top:5px;'>"

        for stat in stats:
            if(stat[0] != ''):
                out += "<span style='margin-left:10px'>-</span><span style='margin-left:10px'><font face='courier' size='2'>"+str(stat[0]) + ": "+str(stat[1])+"</font></span>"
            else:
                out += "<span style='margin-left:10px'>-</span><span style='margin-left:10px'><font face='courier' size='2'>"+str(stat[1])+"</font></span>"

        out += "</div>"
        self.ptext.value = out
        
    def success(self):
        self.pbar.bar_style = 'success'  
        
    def info(self):
        self.pbar.bar_style = 'info'  
        
    def warning(self):
        self.pbar.bar_style = 'warning'  
        
    def danger(self):
        self.pbar.bar_style = 'danger'          
        
    def normal(self):
        self.pbar.bar_style = ''                  


# functionality that doesn't let keyboard interrupt break a process
# it's useful for helping things fail gracefully.
class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)        