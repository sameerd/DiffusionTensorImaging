""" Tracer classes to trace values of various arrays during gradient descent"""
from abc import ABC
import numpy as np

import matplotlib.pyplot as plt

class Tracer(ABC):
    
    """Abstract class to trace the values in the Free Water Gradient Descent"""

    def __init__(self, val_names, split_char=","):
        self.val_list = []
        self.val_names = None
        self.set_val_names(val_names, split_char)
        self.rec = None
        
    def set_val_names(self, names, split_char=","):
        val_names = names
        if isinstance(names, str):
            val_names = names.split(split_char)
        self.val_names = val_names

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

    def check_if_finalized(self):
        if self.rec is None:
            raise ValueError("Tracer is not finalized")

    def plot_separate(self):
        self.check_if_finalized()
        fig, axes = plt.subplots((len(self.val_names) - 1) // 3 + 1, 
                3, sharex=True)
        for i, val in enumerate(self.val_names):
            if len(self.val_names) > 3:
                ax = axes[i // 3, i % 3]
            else:
                ax = axes[i % 3]
            ax.set_title(val)
            ax.plot(self.rec[val])
        plt.tight_layout()

    def plot_overlaid(self):
        self.check_if_finalized()
        fig = plt.figure()
        ax = fig.gca()
        for val in self.val_names:
            ax.plot(self.rec[val], label=val)
            ax.set_xlabel("Iteration")
        plt.legend()


class LossTracer(Tracer):
    """Trace the loss values"""
    
    def __init__(self, val_names="fidelity,beltrami,loss"):
        Tracer.__init__(self, val_names)
        
    def trace_after_loss_functions(self, fw):
        total_loss_fid = np.asscalar(fw.total_loss_fid)
        total_loss_vol = np.asscalar(fw.total_loss_vol)
        loss = np.asscalar(fw.loss)
        self.add_to_list([total_loss_fid, total_loss_vol, loss])  
    
class IdxTracer(Tracer):
    """Trace values at one index"""
        
    def __init__(self, idx, 
            val_names="loss_vol,loss_fid,f,finc,x4inc,detg,x4m"):
        Tracer.__init__(self, val_names)
        self.idx = idx
    
    def trace_after_loss_functions(self, fw):
        def extract_scalar_at_idx(arr_name):
            return(np.asscalar(getattr(fw, arr_name)[self.idx]))
        arr_names = self.val_names[:-1] # remove x4m
        #x4m is extracted differently
        x4m = np.asscalar(fw.manifold[...,0][self.idx])
        self.add_to_list(
            [extract_scalar_at_idx(name) for name in arr_names] + 
            [x4m])

    def plot_overlaid(self):
        Tracer.plot_overlaid(self)
        plt.ylabel("Loss Value")

if __name__ == "__main__":
    # test out the loss tracer
    lt = LossTracer()
    class FW: # create a fake class
        pass
    fw = FW()
    for i in range(20):
        # create some fake numpy arrays
        fw.total_loss_fid = np.array(i ** 2)
        fw.total_loss_vol = np.array(i ** 2 + 3)
        fw.loss = np.sqrt(i)
        # feed the fake values to the loss tracer
        lt.trace_after_loss_functions(fw)
    # finalize tracer and plot
    lt.finalize(fw)
    lt.plot_separate()
