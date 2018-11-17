""" FreewaterRunner interfaces with FreeWaterGradientClass and makes heavy use of DIPY """
import numpy as np
import dipy.reconst.dti as dti
from dipy.reconst.vec_val_sum import vec_val_vect

from freewater import FreeWaterGradientDescent, create_feature_spatial_manifold
from freewater import convert_manifold_to_lower_tri_order
from tracer import LossTracer

MIN_POSITIVE_EIGENVALUE = 1e-12

def log_transform_to_qform(evals, evecs):
    """Takes evals and evecs and returns a q-form that can be sent into free water calc"""
    evals_copy = evals.copy()
    evals_copy[evals_copy <= 0] = MIN_POSITIVE_EIGENVALUE
    qform = vec_val_vect(evecs, evals)
    return(qform)

def exp_transform_from_qform(qform):
    """Takes a q form and returns evals and evecs"""
    return (dti.decompose_tensor(dti.from_lower_triangular(qform), min_diffusivity=MIN_POSITIVE_EIGENVALUE))

class FreewaterRunner:
    """ This class interfaces with the class FreeWaterGradientDescent which actually 
        runs the gradient descent. Here we
          1. Prepare the data in form required by class FreeWaterGradientDescent
          2. Do the log-euclidean transform
          3. Do the reverse log-euclidean transform to get the data back in the manifold"""
    
    LOG = True
    
    MAX_DIFFUSIVITY = 5 * 1e-3
    MIN_DIFFUSIVITY = 0.01 * 1e-3
    
    LAMBDA_MAX = 1 * 1e-3 # max diffusivity in tissue
    LAMBDA_MIN = 0.3 * 1e-3 # min diffusivity in tissue

    # apparent diffusion coefficient in water
    ADC_WATER = 3 * 1e-3


    def __init__(self, data, gtab, xslice=slice(None, None),
                 yslice=slice(None, None), zslice=slice(None, None)):
        # shape of original_data (xs, ys, zs, num_gradients)
        self.original_data = data # grab a reference to the original data
        self.data = data[xslice, yslice, zslice, :].astype(np.float_, copy=True)
        self.gtab = gtab

        self.xslice = xslice
        self.yslice = yslice
        self.zslice = zslice
        
        self.data_b0 = None # The b0 part of the data
        self.Stissue = None # The likely signal intensity of deep tissue. 
        self.Swater = None # The likely signal intesity of water
        self.b_value = None # The b-value of the single shell.
        
        self.Ahat = None # Signal attenuations shape (xs, ys, zs, num_gradients)
        
        # tissue fraction of a voxel
        self.f_init = None # shape (xs, ys, zs, 1) Also, (freewater = 1 - f)
        self.fmin = None # constraints on the values of f. shape (xs, ys, zs)
        self.fmax = None # constraints on the values of f. shape (xs, ys, zs)
        
        self.init_data() 
        self.init_tissuefraction()
        
        self.fw_gd = None # Free water Gradient descent object
        self.loss_tracer = None 
        self.evals = None # free water corrected eigenvalues
        self.fw = None # free water map
        
    def init_data(self):
        # make sure that all small signals or zero or negative are replaced 
        # by the minimum positive signal
        self.data[self.data <= dti.MIN_POSITIVE_SIGNAL] = dti.MIN_POSITIVE_SIGNAL

        # Let's extract out the b0 part of the data
        self.data_b0 = self.data[:,:,:, self.gtab.b0s_mask]

        # If we do not have the signal intensity of tissue and Water in the b0 image
        # Let us estimate it to be `the upper ends of the distribution
        ret = np.percentile(self.data_b0.ravel(), q=[0.1, 99.9])
        if self.Stissue is None: self.Stissue = ret[0]
        if self.Swater is None: self.Swater = ret[1]
        if self.LOG: print("Stissue =", self.Stissue, ": Swater =", self.Swater)

        # Set the single shell bval to be the first index in the gradient table that is not marked as 
        # as a b0 value. All these values should be the same (eg. b=1000) however sometimes we have values
        # that are close to 1000 (eg. 995, 1005 etc). We are just picking the first non-zero value and 
        # that should be good enough
        self.b_value = self.gtab.bvals[np.where(~self.gtab.b0s_mask)[0][0]]
        if self.LOG: print("Single shell bvalue =", self.b_value)
        
        # calculate the signal attenuations
        self.Ahat = self.data / self.data_b0

        attenuation_min = np.exp(- self.b_value * self.MAX_DIFFUSIVITY )
        attenuation_max = np.exp(- self.b_value * self.MIN_DIFFUSIVITY)

        if self.LOG:
            print("Attenuation_min = %.3f, Attenuation_max = %.3f" % 
                  (attenuation_min, attenuation_max))

        #Clip the attenuations to attenuation_min and attenuation_max
        mask = self.Ahat < attenuation_min
        mask[:,:,:, self.gtab.b0s_mask] = False
        self.Ahat[self.Ahat < attenuation_min] = attenuation_min
        
        mask = self.Ahat > attenuation_max
        mask[:,:,:, self.gtab.b0s_mask] = False
        self.Ahat[self.Ahat > attenuation_max] = attenuation_max

        if self.LOG: print("Ahat.shape =", self.Ahat.shape)

            
    def init_tissuefraction(self):
        
        # calculate Initial tissue fraction (1 - freewater)
        Awater_scalar = np.exp(- self.b_value * self.ADC_WATER)
        if self.LOG: print("Awater_k = %.2f" % Awater_scalar)

        # Max and min attenuation. (Amax uses lambda_min and vice versa)
        Amax = np.exp(-self.b_value * self.LAMBDA_MIN)
        Amin = np.exp(-self.b_value * self.LAMBDA_MAX)

        Ahat_min = self.Ahat[..., ~self.gtab.b0s_mask].min(axis=3)
        Ahat_max = self.Ahat[..., ~self.gtab.b0s_mask].max(axis=3)

        # Note this formula is perhaps a correction of Equation [6]
        self.fmin = (Ahat_min - Awater_scalar) / (Amax - Awater_scalar)
        self.fmax = (Ahat_max - Awater_scalar) / (Amin - Awater_scalar)

        # now make sure fmax and fmin are between 0 and 1
        self.fmin[self.fmin <= 0] = 0.01
        self.fmin[self.fmin >= 1] = 0.99
        self.fmax[self.fmax <= 0] = 0.01
        self.fmax[self.fmax >= 1] = 0.99

        f_init = 1 - (np.log(np.squeeze(self.data_b0) / self.Stissue) / np.log(self.Swater / self.Stissue))
        mask = f_init < self.fmin
        f_init[mask] = (self.fmin[mask] + self.fmax[mask])/2
        mask = f_init > self.fmax
        f_init[mask] = (self.fmin[mask] + self.fmax[mask])/2

        # This is our starting value of f (the tissue fraction)
        self.f_init = f_init[:,:,:,np.newaxis]

        if self.LOG: print("f_init.shape =", self.f_init.shape)
            
    def run_model(self, num_iter=100, dt = 0.001, beta = 100.):
        
        # first we fit a single tensor model 
        tenmodel = dti.TensorModel(self.gtab)
        tenfit = tenmodel.fit(self.data)

        # Then we feed the output of the signal tensor model into the bi-tensor model
        # by first doing the log euclidean transform
        qform = log_transform_to_qform(tenfit.evals, tenfit.evecs)
        manifold = create_feature_spatial_manifold(self.data_b0, qform)
        num_gradients = np.sum(~self.gtab.b0s_mask)
        
        # scale down the timestep
        dt_scaled = dt / (self.b_value * num_gradients)

        
        self.fw_gd = FreeWaterGradientDescent(self.Ahat, manifold, self.f_init, 
                                              self.gtab, self.b_value, dt_scaled, 
                                              fmin=self.fmin, fmax=self.fmax )

        self.fw_gd.beta = beta
        
        self.loss_tracer = LossTracer()
        self.fw_gd.init_tracers([self.loss_tracer])

        for i in range(num_iter):
            self.fw_gd.iterate()
        self.fw_gd.finalize()
        
        # Now do the inverse of the log euclidean transform to bring the values back to 
        # our manifold
        ret_qform = convert_manifold_to_lower_tri_order(self.fw_gd.manifold)
        self.evals, evecs = exp_transform_from_qform(ret_qform)
        self.fw = 1 - self.fw_gd.f # freewater = 1 - tissue_fraction

    def plot_loss(self, figsize=(10,4)):
        if self.loss_tracer is not None:
            self.loss_tracer.plot_separate(figsize=(10, 4))

    def _convert_array_to_original_self(self, arr):
        """ arr is an array of small size (shape = (xslice, yslice, zslice). 
            Convert it to the original size and return (shape = self.original_data)
        """
        ret = np.zeros(self.original_data.shape[:3]) # default dtype is float
        ret[self.xslice, self.yslice, self.zslice] = np.squeeze(arr)
        return ret
            
    def get_fw_map(self):
        """ Free water map"""
        return self._convert_array_to_original_self(self.fw)
    
    def get_fw_md(self):
        """ Free water corrected mean diffusivity"""
        if self.evals is not None:
            return self._convert_array_to_original_self(
                        self.evals.mean(axis=-1))
        
    def get_fw_fa(self):
        """ Free water corrected fractional anisotropy"""
        if self.evals is not None:
            return self._convert_array_to_original_self(
                        dti.fractional_anisotropy(self.evals))


