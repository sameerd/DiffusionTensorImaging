""" Tracer classes to trace values of various arrays during gradient descent"""
from abc import ABC
import numpy as np

import matplotlib.pyplot as plt

def functor_extract_scalar_at_idx(idx):
    def extract_scalar_at_idx(fw, arr_name):
        return(np.asscalar(getattr(fw, arr_name)[idx]))
    return(extract_scalar_at_idx)

def extract_scalar(fw, name):
    return(np.asscalar(getattr(fw, name)))

def functor_extract_scalar_from_manifold_at_idx(man_idx, idx):
    """Creates a function that will extract an index in the manifold"""
    def extract_scalar_from_manifold_at_idx(fw, name):
        return(np.asscalar(fw.manifold[...,man_idx][idx]))
    return(extract_scalar_from_manifold_at_idx)


class Tracer(ABC):
    
    """Abstract class to trace the values in the Free Water Gradient Descent"""

    def __init__(self, val_names, split_char=","):
        self.val_list = []
        self.val_names = self.guess_val_names(val_names, split_char)
        func = self.get_default_extract_func()
        self.extract_funcd = {name:func for name in self.val_names}
        self.rec = None

    def get_default_extract_func(self):
        """ Get default extractor funcs. Overridden in derived classes """
        return(extract_scalar)

    def set_extract_func(self, val_name, func):
        self.extract_funcd[val_name] = func
        
    def guess_val_names(self, names, split_char=","):
        """If names is a string them split it on split_char, 
            else it is a list of strings"""
        val_names = names
        if isinstance(names, str):
            val_names = names.split(split_char)
        return(val_names)

    def trace(self, location, fw):
        """Try to call the function named trace + location """
        func_name = "trace" + location
        return (getattr(self, func_name)(fw))

    def trace_after_increments_compute(self, fw):
        pass
    
    def trace_after_loss_functions(self, fw):
        pass
    
    def trace_after_increment(self, fw):
        pass
    
    def trace_after_constrain(self, fw):
        pass
    
    def add_vals_to_list(self, fw):
        """ Call all the extractor functions """
        vals = [self.extract_funcd[name](fw, name) for name in self.val_names]
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
    
    def __init__(self, val_names="loss,total_loss_fid,total_loss_vol"):
        Tracer.__init__(self, val_names)

    def get_default_extract_func(self):
        return(extract_scalar)
        
    def trace_after_loss_functions(self, fw):
        self.add_vals_to_list(fw)


class IdxTracer(Tracer):
    """Trace values at one index"""
        
    def __init__(self, idx, 
            val_names="loss_vol,loss_fid,f,finc,x4inc,detg,x4m"):
        self.idx = idx
        Tracer.__init__(self, val_names)
        if "x4m" in self.val_names:
            self.set_extract_func("x4m", 
                functor_extract_scalar_from_manifold_at_idx(0, self.idx))

    def get_default_extract_func(self):
        return(functor_extract_scalar_at_idx(self.idx))
    
    def trace_after_loss_functions(self, fw):
        self.add_vals_to_list(fw)

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
