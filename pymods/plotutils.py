import matplotlib.pyplot as plt

def plot_3d_z_indices(vol, idx_list, transpose=True,
                      num_rows=1, num_cols=1, 
                      label_loc_x=5, label_loc_y=5, 
                      im_origin = "lower"):
    """Plot z-index slices of a 3d volume as subplots"""
    vol = vol.squeeze() # if there are extra dimensions then reduce them
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    fig.subplots_adjust(wspace=0.025, hspace=0.0)

    im = None
    for i, idx in enumerate(idx_list):
        plt.subplot(num_rows, num_cols, i+1)
        plt.text(label_loc_x, label_loc_y, str(idx), fontsize=18, 
                 ha="center", color="black",
                 bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        if transpose is True:
            im = plt.imshow(vol[:,:,idx].transpose(), origin=im_origin)
        else:
            im = plt.imshow(vol[:,:,idx], origin=im_origin)

    # now put a color bar on the right
    fig = plt.gcf()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)

