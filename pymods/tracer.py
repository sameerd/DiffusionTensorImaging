""" Tracer classes to trace values of various arrays during gradient descent"""
from abc import ABC
import numpy as np

class Tracer(ABC):
    
    """Abstract class to trace the values in the Free Water Gradient Descent"""

    def __init__(self):
        self.val_list = []
        self.val_names = ""
        self.rec = None
        
    def set_val_names(self, names):
        self.val_names = names
        
    def trace_after_increments_compute(self, fw):
        pass
    
    def trace_after_loss_functions(self, fw):
        pass
    
    def trace_after_increment(self, fw):
        pass
    
    def trace_after_constrain(self, fw):
        pass
    
    def add_to_list(self, vals):
        self.val_list.append(vals)
    
    def finalize(self, fw):
        if self.val_list:
            self.rec = np.rec.fromrecords(self.val_list, names=self.val_names)

class LossTracer(Tracer):
    """Trace the loss values"""
    
    def __init__(self):
        Tracer.__init__(self)
        self.set_val_names("fidelity,beltrami,loss")
        
    def trace_after_loss_functions(self, fw):
        total_loss_fid = np.asscalar(fw.total_loss_fid)
        total_loss_vol = np.asscalar(fw.total_loss_vol)
        loss = np.asscalar(fw.loss)
        self.add_to_list([total_loss_fid, total_loss_vol, loss])  
    
class IdxTracer(Tracer):
    """Trace values at one index"""
        
    def __init__(self, idx):
        Tracer.__init__(self)
        self.idx = idx
        self.set_val_names("loss,f,finc,x4inc,detg,x4m")
    
    def trace_after_loss_functions(self, fw):
        def extract_scalar_at_idx(arr_name):
            return(np.asscalar(getattr(fw, arr_name)[self.idx]))
        arr_names = ["loss_fid", "f", "finc", "x4inc", "detg"]
        x4m = np.asscalar(fw.manifold[...,0][idx])
        self.add_to_list([extract_scalar_at_idx(name) for name in arr_names] + 
                [x4m])
