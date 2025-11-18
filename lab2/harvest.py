# -*- coding: utf-8 -*-

#%% packages used by this file
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#%% Harvest funktionen
def harvest(n, p, mu, y_cond=None):
    """
    Plot densities for binomial sum of Poisson
    
    Plots the joint density for the number of harvested seeds (Y) and the
    number of original seeds that grew (X); the marginal density for the 
    number of harvested seeds (Y); and, if y is given, the conditional 
    density for the number of original seeds that grew (X|Y=y).
    
    The model is
      X ~ Bin(n,p)
    and, assuming X=k,
      Y = sum_i^k Z_i ~ Po(k*mu)
    where the number of harvested seeds for each original seed that grew is
      Z_i ~ Po(mu)

    Parameters
    ----------
    n : Positive integer
        Number of seeds to consider
    p : Number in 0 to 1
        Probability of growth
    mu : Positive number
         Mean value of yield for each seed that grows.
    y_cond : Positive number, optional
        Observed yield. If given, also compute conditional distribution for 
        the number of seeds that grew, given yield.
        The default is None.

    Returns
    -------
    Figure object of matplotlib.figure containing plots of the densities.
    
    Johan Lindström
    """
    # Determin Ymax, based on E(Y) and V(Y)
    tot_E = n*p*mu
    tot_V = tot_E*(1+mu*(1-p))
    Ymax = tot_E+4*np.sqrt(tot_V)
    # ensure that Ymax is as large as y_cond
    if not (y_cond is None):
        Ymax = max([Ymax,y_cond])
    
    #create vectors for x and y
    x = np.arange(n+1)
    y = np.arange(Ymax+1)
    # expand vector to grid
    X_in,Y_in = np.meshgrid(x, y)
    
    # Compute joint probabilities using Law of Total Prob.
    pXY = stats.binom.pmf(X_in,n,p) * stats.poisson.pmf(Y_in,X_in*mu)
    # Compute marginal probability
    pY = pXY.sum(axis=1)
    
    if y_cond is None:
      pX_Y = None
    else:
      #conditional p_X|Y
      #what should be computed
      # pX_Y = stats.binom.pmf(x,n,p)*stats.poisson.pmf(y_cond,x*mu)
      #However, we use a numerically stable log approach
      pX_Y = np.log(stats.binom.pmf(x,n,p)) - x*mu
      #we also need to add (x*mu)^y (but not for x[0]=0)
      pX_Y[1:] = pX_Y[1:] + y_cond*np.log(x[1:]*mu)
      #first point should instead be log(0)=-Inf (0 when we transform back)
      pX_Y[0] = -np.Inf
      #convert to standard scale
      pX_Y = np.exp(pX_Y-max(pX_Y))
      #and normalize
      pX_Y = pX_Y / sum(pX_Y)
    #if y_cond is None; else
     
    fig = plt.figure()
    ax = [[fig.add_subplot(2, 2, 1, projection='3d'),
           fig.add_subplot(2, 2, 2)],
          [fig.add_subplot(2, 2, 3),None]]
    
    #justera plottar så att texten inte överlappar
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    
    ax = np.array(ax)
    ax[0,0].stem(X_in.flatten(), Y_in.flatten(), pXY.flatten(), 
                 basefmt=' ', markerfmt=' ')
    ax[0,0].set_xlabel('k')
    ax[0,0].set_ylabel('l')
    ax[0,0].set_title('$p_{XY}(k,l)$')
        
    ax[0,1].pcolormesh(x,y,pXY)
    ax[0,1].set_xlabel('k')
    ax[0,1].set_ylabel('l')
    ax[0,1].set_title('$p_{XY}(k,l)$')
    
    ax[1,0].fill_between(y, pY)
    ax[1,0].set_title('Marginaltäthet för nya frön')
    ax[1,0].set_xlabel('antal nya frön (l)')
    ax[1,0].set_ylabel('$p_Y(l)$')
    
    if not (y_cond is None):
        ax[1,1] = fig.add_subplot(2, 2, 4)
        ax[1,1].bar(x, pX_Y)
        ax[1,1].set_xlabel('antal som gror (k)')
        ax[1,1].set_ylabel('$p_{X|Y=' + str(y_cond) + '}(k)$')
        ax[1,1].set_title('Betingadtäthet')
        #also add conditioning line to the joint density
        ax[0,0].plot((0,n), (y_cond,y_cond), (0,0), color='red')
        ax[0,1].axline((0,y_cond), (n,y_cond), color='red')
    #if not (y_cond is None):
    
    ##return
    return fig
