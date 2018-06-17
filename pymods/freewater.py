import numpy as np

def dx_manifold(manifold, idx):
    "Backwards difference of manifold co-ordinate idx along the x-axis"
    dx = np.empty(manifold.shape[:-1] + (1,), dtype=manifold.dtype)
    dx[1:,:, :, 0] = manifold[1:, :, :, idx] - manifold[:-1,:, :, idx]
    dx[0, :, :, 0] = 0.
    return(dx)

def dy_manifold(manifold, idx):
    "Backwards difference of manifold co-ordinate idx along the y-axis"
    dy = np.empty(manifold.shape[:-1] + (1,), dtype=manifold.dtype)
    dy[:,1:, :, 0] = manifold[:, 1:, :, idx] - manifold[:, :-1, :, idx]
    dy[:, 0, :, 0] = 0.
    return(dy)

def dz_manifold(manifold, idx):
    "Backwards difference of manifold co-ordinate idx along the z-axis"
    dz = np.empty(manifold.shape[:-1] + (1,), dtype=manifold.dtype)
    dz[:, :,1:, 0] = manifold[:,:, 1:, idx] - manifold[:, :, :-1, idx]
    dz[:, :,0, 0] = 0.
    return(dz)

def dpx(p):
    """Forwards derivative of the p,q,r vectors wrt x"""
    p = np.squeeze(p)
    ret = np.empty_like(p)
    ret[:-1, :, :] = p[1:, :, :] - p[:-1, :,:]
    ret[-1, :, :] = 0.
    return(ret)

def dpy(p):
    """Forwards derivative of the p,q,r vectors wrt y"""
    p = np.squeeze(p)
    ret = np.empty_like(p)
    ret[:, :-1, :] = p[:,1:, :] - p[:, :-1, :]
    ret[:, -1, :] = 0.
    return(ret)

def dpz(p):
    """Forwards derivative of the p,q,r vectors wrt z"""
    p = np.squeeze(p)
    ret = np.empty_like(p)
    ret[:,:, :-1] = p[:, :, 1:] - p[:, :, :-1]
    ret[:,:, -1] = 0.
    return(ret)

def create_feature_spatial_manifold(S0, D):
    b0_shape = S0.shape
    d_shape = D.shape
    if (b0_shape[:3] != d_shape[:3]):
        raise ValueError("b0_shape = %s and d_shape = %s." \
                         "The first three dimensions need to match" 
                         % (b0_shape, d_shape))
    manifold_shape = d_shape[:3] + (6,)
    manifold = np.empty(manifold_shape, dtype=D.dtype)
    manifold[...,0] = D[...,0,0]
    manifold[...,1] = D[...,1,1]
    manifold[...,2] = D[...,2,2]
    manifold[...,3] = D[...,0,1] * np.sqrt(2)
    manifold[...,4] = D[...,1,2] * np.sqrt(2)
    manifold[...,5] = D[...,0,2] * np.sqrt(2)
    return(manifold)

class FreeWaterGradientDescent:
    
    """Class to run the Free Water Gradient Descent Computations"""
    
    UNDERFLOW_MIN = 1e-20
    
    def __init__(self, Ahat, manifold, f_init, gtab, b_value, dt, 
                 beta = 10., alpha_reg=1., water_d = 3e-3, 
                 fmin = 0., fmax = 1.):
        self.b_value = b_value
        self.dt = dt
        
        self.beta = beta 
        self.alpha_reg = alpha_reg
        # internal constants that do not change anything 
        # (can be used to modify flow)
        self.alpha_fid = 1. 
        self.alpha_f = 1.
        
        self.Awater_scalar = np.exp(-water_d * self.b_value)
        
        # these two variables are not modified at all in this class
        self.gtab = gtab
        self.Ahat = Ahat
        self.qk = self.gtab.bvecs[~self.gtab.b0s_mask]
        
        self.manifold_init = manifold
        self.f_init = f_init
        self.fmin = fmin
        self.fmax = fmax
        # numpy arrays that will be iteratated over
        self.manifold = self.manifold_init.copy()
        self.f = self.f_init.copy()
        
        # All the intermediate arrays should be initialized however lets not do it right now
        self.tracers = []
        
    def compute_regularization_increments(self):
        self.X4x = dx_manifold(self.manifold, 0)
        self.X4y = dy_manifold(self.manifold, 0)
        self.X4z = dz_manifold(self.manifold, 0)

        self.X5x = dx_manifold(self.manifold, 1)
        self.X5y = dy_manifold(self.manifold, 1)
        self.X5z = dz_manifold(self.manifold, 1)

        self.X6x = dx_manifold(self.manifold, 2)
        self.X6y = dy_manifold(self.manifold, 2)
        self.X6z = dz_manifold(self.manifold, 2)

        self.X7x = dx_manifold(self.manifold, 3)
        self.X7y = dy_manifold(self.manifold, 3)
        self.X7z = dz_manifold(self.manifold, 3)

        self.X8x = dx_manifold(self.manifold, 4)
        self.X8y = dy_manifold(self.manifold, 4)
        self.X8z = dz_manifold(self.manifold, 4)

        self.X9x = dx_manifold(self.manifold, 5)
        self.X9y = dy_manifold(self.manifold, 5)
        self.X9z = dz_manifold(self.manifold, 5)
        
        self.g11 = np.ones(self.X4x.shape) + self.beta * \
            (self.X4x*self.X4x + self.X5x*self.X5x + self.X6x*self.X6x + self.X7x*self.X7x + 
                                       self.X8x*self.X8x + self.X9x*self.X9x)
        self.g22 = np.ones(self.X4y.shape) + self.beta * \
            (self.X4y*self.X4y + self.X5y*self.X5y + self.X6y*self.X6y + self.X7y*self.X7y + 
                                       self.X8y*self.X8y + self.X9y*self.X9y)
        self.g33 = np.ones(self.X4z.shape) + self.beta * \
            (self.X4z*self.X4z + self.X5z*self.X5z + self.X6z*self.X6z + self.X7z*self.X7z + 
                                       self.X8z*self.X8z + self.X9z*self.X9z)
        self.g12 = self.beta * (self.X4x*self.X4y + self.X5x*self.X5y + self.X6x*self.X6y + 
                                self.X7x*self.X7y + self.X8x*self.X8y + self.X9x*self.X9y)
        self.g23 = self.beta * (self.X4y*self.X4z + self.X5y*self.X5z + self.X6y*self.X6z + 
                                self.X7y*self.X7z + self.X8y*self.X8z + self.X9y*self.X9z)
        self.g13 = self.beta * (self.X4x*self.X4z + self.X5x*self.X5z + self.X6x*self.X6z + 
                                self.X7x*self.X7z + self.X8x*self.X8z + self.X9x*self.X9z)
    
        self.C11 = self.g22*self.g33 - self.g23*self.g23
        self.C22 = self.g11*self.g33 - self.g13*self.g13
        self.C33 = self.g11*self.g22 - self.g12*self.g12
        self.C12 = -self.g12*self.g33 + self.g13*self.g23
        self.C23 = -self.g11*self.g23 + self.g13*self.g12
        self.C13 = self.g12*self.g23 - self.g13*self.g22
    
        self.detg = self.g11*self.C11 + self.g12*self.C12 + self.g13*self.C13

        self.gm05 = 1. / np.sqrt(self.detg)

        self.p4 = self.C11*self.X4x + self.C12*self.X4y + self.C13*self.X4z
        self.p5 = self.C11*self.X5x + self.C12*self.X5y + self.C13*self.X5z
        self.p6 = self.C11*self.X6x + self.C12*self.X6y + self.C13*self.X6z
        self.p7 = self.C11*self.X7x + self.C12*self.X7y + self.C13*self.X7z
        self.p8 = self.C11*self.X8x + self.C12*self.X8y + self.C13*self.X8z
        self.p9 = self.C11*self.X9x + self.C12*self.X9y + self.C13*self.X9z

        self.q4 = self.C12*self.X4x + self.C22*self.X4y + self.C23*self.X4z
        self.q5 = self.C12*self.X5x + self.C22*self.X5y + self.C23*self.X5z
        self.q6 = self.C12*self.X6x + self.C22*self.X6y + self.C23*self.X6z
        self.q7 = self.C12*self.X7x + self.C22*self.X7y + self.C23*self.X7z
        self.q8 = self.C12*self.X8x + self.C22*self.X8y + self.C23*self.X8z
        self.q9 = self.C12*self.X9x + self.C22*self.X9y + self.C23*self.X9z

        self.r4 = self.C13*self.X4x + self.C23*self.X4y + self.C33*self.X4z
        self.r5 = self.C13*self.X5x + self.C23*self.X5y + self.C33*self.X5z
        self.r6 = self.C13*self.X6x + self.C23*self.X6y + self.C33*self.X6z
        self.r7 = self.C13*self.X7x + self.C23*self.X7y + self.C33*self.X7z
        self.r8 = self.C13*self.X8x + self.C23*self.X8y + self.C33*self.X8z
        self.r9 = self.C13*self.X9x + self.C23*self.X9y + self.C33*self.X9z

        # Beltrami operator incrementals
        self.b4inc = (dpx(self.p4 * self.gm05) + dpy(self.q4 * self.gm05) + dpz(self.r4 * self.gm05)) * np.squeeze(self.gm05)
        self.b5inc = (dpx(self.p5 * self.gm05) + dpy(self.q5 * self.gm05) + dpz(self.r5 * self.gm05)) * np.squeeze(self.gm05)
        self.b6inc = (dpx(self.p6 * self.gm05) + dpy(self.q6 * self.gm05) + dpz(self.r6 * self.gm05)) * np.squeeze(self.gm05)
        self.b7inc = (dpx(self.p7 * self.gm05) + dpy(self.q7 * self.gm05) + dpz(self.r7 * self.gm05)) * np.squeeze(self.gm05)
        self.b8inc = (dpx(self.p8 * self.gm05) + dpy(self.q8 * self.gm05) + dpz(self.r8 * self.gm05)) * np.squeeze(self.gm05)
        self.b9inc = (dpx(self.p9 * self.gm05) + dpy(self.q9 * self.gm05) + dpz(self.r9 * self.gm05)) * np.squeeze(self.gm05)


    def compute_fidelity_increments(self):

        self.Ahat_tissue_curr = \
            self.qk[:, 0] * self.qk[:, 0] * self.manifold[..., 0:1] + \
            self.qk[:, 1] * self.qk[:, 1] * self.manifold[..., 1:2] + \
            self.qk[:, 2] * self.qk[:, 2] * self.manifold[..., 2:3] + \
            self.qk[:, 0] * self.qk[:, 1] * self.manifold[..., 3:4] * np.sqrt(2) + \
            self.qk[:, 1] * self.qk[:, 2] * self.manifold[..., 4:5] * np.sqrt(2) + \
            self.qk[:, 0] * self.qk[:, 2] * self.manifold[..., 5:6] * np.sqrt(2)
        # prevent underflow
        np.clip(self.Ahat_tissue_curr, a_min=1e-7, a_max=None, out=self.Ahat_tissue_curr) 
        self.Ahat_tissue_curr = np.exp(-self.b_value * self.Ahat_tissue_curr)

        self.A_bi = self.f * self.Ahat_tissue_curr + (1 - self.f) * self.Awater_scalar

        self.fidmat = self.b_value * (self.Ahat[..., ~self.gtab.b0s_mask] - self.A_bi) * self.Ahat_tissue_curr
        # FIXME: Do we multiply by f or not? f is mostly positive so it might not matter.
        self.fidmat = self.f * self.fidmat
        
        self.fid4inc = (self.fidmat * (self.qk[:, 0] * self.qk[:, 0])).sum(axis=-1)
        self.fid5inc = (self.fidmat * (self.qk[:, 1] * self.qk[:, 1])).sum(axis=-1)
        self.fid6inc = (self.fidmat * (self.qk[:, 2] * self.qk[:, 2])).sum(axis=-1)
        self.fid7inc = (self.fidmat * (self.qk[:, 0] * self.qk[:, 1])).sum(axis=-1) * np.sqrt(2) # 2 / sqrt(2)
        self.fid8inc = (self.fidmat * (self.qk[:, 1] * self.qk[:, 2])).sum(axis=-1) * np.sqrt(2)
        self.fid9inc = (self.fidmat * (self.qk[:, 0] * self.qk[:, 2])).sum(axis=-1) * np.sqrt(2)

    def compute_f_increment(self):
        self.finc = (self.alpha_f) * (-self.b_value) * (
            (self.Ahat[..., ~self.gtab.b0s_mask] - self.A_bi) * 
            (self.Ahat_tissue_curr - self.Awater_scalar)).sum(axis=-1)

    def compute_manifold_increments(self):
        #incrementals are the sum of the fidelity incrementals and the beltrami incrementals
        def compute_one_manifold_increment(fidinc, reginc):
            return(self.alpha_fid * fidinc + self.alpha_reg * reginc)

        self.x4inc = compute_one_manifold_increment(self.fid4inc, self.b4inc)
        self.x5inc = compute_one_manifold_increment(self.fid5inc, self.b5inc)
        self.x6inc = compute_one_manifold_increment(self.fid6inc, self.b6inc)
        self.x7inc = compute_one_manifold_increment(self.fid7inc, self.b7inc)
        self.x8inc = compute_one_manifold_increment(self.fid8inc, self.b8inc)
        self.x9inc = compute_one_manifold_increment(self.fid9inc, self.b9inc)


    def compute_fidelity_loss(self):
        self.loss_fid = np.linalg.norm(self.Ahat[..., ~self.gtab.b0s_mask] - self.A_bi, axis=-1)

    def compute_volume_loss(self):
        self.loss_vol = np.sqrt(self.detg)

    def compute_total_fidelity_loss(self):
        self.total_loss_fid = np.sum(self.loss_fid)

    def compute_total_volume_loss(self):
        self.total_loss_vol = np.sum(self.loss_vol)

    def compute_total_loss(self):
        self.loss = self.alpha_fid*self.total_loss_fid + self.alpha_reg*self.total_loss_vol

    def increment_manifold(self):
        self.manifold[...,0] += self.dt * self.x4inc
        self.manifold[...,1] += self.dt * self.x5inc
        self.manifold[...,2] += self.dt * self.x6inc
        self.manifold[...,3] += self.dt * self.x7inc
        self.manifold[...,4] += self.dt * self.x8inc
        self.manifold[...,5] += self.dt * self.x9inc

    def constrain_manifold(self):
        #np.clip(self.manifold, a_min=self.UNDERFLOW_MIN, a_max=1., out=self.manifold)
        np.clip(self.manifold, a_min=-1., a_max=1., out=self.manifold)

    def increment_f(self):
        self.f[...,0] +=  self.dt * self.finc
        self.f = self.f.squeeze()

    def constrain_f(self):
        # make sure that the f values stay constrained
        np.clip(self.f, a_min=self.fmin, a_max=self.fmax, out=self.f)
        self.f = self.f[...,np.newaxis]

    def iterate(self):
        # compute increments
        self.compute_regularization_increments()
        self.compute_fidelity_increments()
        self.compute_manifold_increments()
        self.compute_f_increment()
        
        self.trace_after_increments_compute()

        # loss functions
        self.compute_fidelity_loss()
        self.compute_volume_loss()
        self.compute_total_fidelity_loss()
        self.compute_total_volume_loss()
        self.compute_total_loss()

        self.trace_after_loss_functions()
        
        # increment
        self.increment_manifold()
        self.increment_f()
        
        self.trace_after_increment()

        # constrain
        self.constrain_manifold()
        self.constrain_f()
        
        self.trace_after_constrain()
        
    def init_tracers(self, tracers):
        self.tracers = tracers
        
    def trace_after_increments_compute(self):
        for tracer in self.tracers:
            tracer.trace_after_increments_compute(self)
    
    def trace_after_loss_functions(self):
        for tracer in self.tracers:
            tracer.trace_after_loss_functions(self)
    
    def trace_after_increment(self):
        for tracer in self.tracers:
            tracer.trace_after_increment(self)
    
    def trace_after_constrain(self):
        for tracer in self.tracers:
            tracer.trace_after_constrain(self)
            
    def finalize(self):
        for tracer in self.tracers:
            tracer.finalize(self)
