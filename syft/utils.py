from IPython.display import display
from IPython.html.widgets import IntProgress, HBox, HTML

class Progress(object):
    
    def __init__(self,min,max):

        self.pbar = IntProgress(orientation='horizontal',min=0, max=100)
        self.ptext = HTML()
        # Only way to place text to the right of the bar is to use a container
        container = HBox(children=[self.pbar, self.ptext])
        display(container)
    
    def update(self,value,stats):
        self.pbar.value = value
        out = "<div style='margin-left:5px; margin-top:5px;'>"

        for stat in stats:
            if(stat[0] != ''):
                out += "<span style='margin-left:10px'>-</span><span style='margin-left:10px'><font face='courier' size='2'>"+stat[0] + ": "+stat[1]+"</font></span>"
            else:
                out += "<span style='margin-left:10px'>-</span><span style='margin-left:10px'><font face='courier' size='2'>"+stat[1]+"</font></span>"

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