import torch

def accuracy(predictions,dataset):
    """Evaluates accuracy for given set of predictions and true labels.
       Args:
           
           predictions:
           labels:
               
       Returns:
           
           accuracy:
    
    """
    

    
    total=0.0
    correct=0.0
    
    for j in range(0,len(dataset)):
        
        correct+=(predictions[j].long()==dataset[j].long()).sum().item()
               
        total+=len(dataset[j])
        
    return((correct/total)*100)

def plot(x,y):
    """Plots a graph of given x and y.
       Args:
           
           x:
           y:
    """
    pass

def histogram(x,y):
    """Plots a histogram for corresponding x and y:
       Args:
           
           x:
           y:
    """
       
    
