import numpy as np                                        # a fundamental numerical linear algebra library
import matplotlib.pyplot as plt                           # a basic plotting library


# simple 2 panel plotting function - to show data and data + true function in separate panels
def cust_plt_util(data_x,data_y,labels,true_x,true_y):
    # distinguish labels
    pos_inds = np.argwhere(labels == 1)
    pos_inds = [s[0] for s in pos_inds]
    neg_inds = np.argwhere(labels ==-1)
    neg_inds = [s[0] for s in neg_inds]
    
    # setup plot - left panel just data, right panel data + true func
    fig = plt.figure(figsize = (16,5))
    
    ## plot just data
    ax = fig.add_subplot(1,2,1)
    ax.scatter(data_x[pos_inds],data_y[pos_inds],color = 'salmon',linewidth = 1)
    ax.scatter(data_x[neg_inds],data_y[neg_inds],color = 'cornflowerblue',linewidth = 1)
    ax.set_xlim(min(data_x)-0.1,max(data_x)+0.1)
    ax.set_ylim(min(data_y)-0.1,max(data_y)+0.1)
    ax.set_yticks([],[])
    ax.axis('off') 

    ## plot data + true func
    ax = fig.add_subplot(1,2,2)
    ax.plot(true_x,true_y,color = 'k',linestyle = '--',linewidth = 2.5)
    ax.scatter(data_x[pos_inds],data_y[pos_inds],color = 'salmon',linewidth = 1)
    ax.scatter(data_x[neg_inds],data_y[neg_inds],color = 'cornflowerblue',linewidth = 1)
    ax.set_xlim(min(data_x)-0.1,max(data_x)+0.1)
    ax.set_ylim(min(data_y)-0.1,max(data_y)+0.1)
    ax.set_yticks([],[])
    ax.axis('off') 
    
    
# function - plot data with underlying target function generated in the previous Python cell
def plot_data(data_x,data_y,labels,sep_x,sep_y):
    # plot data 
    fig = plt.figure(figsize = (4,4))
    ax = fig.add_subplot(111)
    pos_inds = np.argwhere(labels == 1)
    pos_inds = [s[0] for s in pos_inds]
    neg_inds = np.argwhere(labels ==-1)
    neg_inds = [s[0] for s in neg_inds]
    ax.scatter(data_x[pos_inds],data_y[pos_inds],color = 'salmon',linewidth = 1)
    ax.scatter(data_x[neg_inds],data_y[neg_inds],color = 'cornflowerblue',linewidth = 1)
    
    # plot target
    ax.plot(sep_x,sep_y,color = 'k',linestyle = '--',linewidth = 2.5)
    # clean up plot
    ax.set_yticks([],[])
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])
    ax.axis('off') 
    
# plot approximation
def plot_approx(clf):
    
    # plot classification boundary and color regions appropriately
    r = np.linspace(-2.1,2.1,700)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    # use classifier to make predictions
    z = clf.predict(h)

    # reshape predictions for plotting
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    # show the filled in predicted-regions of the plane 
    plt.contourf(s,t,z,colors = ['cornflowerblue','salmon'],alpha = 0.2,levels = range(-1,2))

    # show the classification boundary if it exists
    if len(np.unique(z)) > 1:
        plt.contour(s,t,z,colors = 'k',linewidths = 2)