# -*- coding: utf-8 -*-
#%% packages used by this file
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#%% funktionen skattningar
def styrkefkn(mu0, sigma, n, alpha=0.05, riktning='!=', mu_sant=None):
	"""
	styrkefkn Illustrerar styrekfunktion for hypotestest
	
	Illustrerar kritiskt omrade och styrekfunktion for hypotestest av mu under
	antagande om n observationer fran en normalfordelning, N(mu_0, sigma).

	Parameters
	----------
	mu0 : real number
		Väntevärde i normalfördelningen.
	sigma : positive number
		Standardavvikelse i normalfördelningen.
	n : positive integer
		Antal observationer.
	alpha : värde mellan 0 och 1
		Testest signifikansnivå. The default is 0.05.
	riktning : Textsträng, optional
		Riktning på testet, '<', '>', eller '!='. The default is '!='.
	mu_sant : real number, optional
		Värde under H1 som styrkan räknas utför och typ I och II fel 
		illustreras. Om det inte anges illustrerars bara styrkan. 
		The default is None.

	Returns
	-------
    Figure object of matplotlib.figure containing plots of the densities.
    
    Johan Lindström
	"""
	#compute standard deviation of estimate
	s = sigma/np.sqrt(n)

	#Beräkna styrkefunktionen för de olika fallen
	match riktning:
		case "<":
			k1 = stats.norm.ppf(alpha, mu0, s)
			k2 = np.Inf
			x = np.linspace(k1-4*s, mu0+3*s, 1000)
		case ">":
			k1 = -np.Inf
			k2 = stats.norm.ppf(1-alpha, mu0, s)
			x = np.linspace(mu0-3*s, k2+4*s, 1000)
		case "!=":
			k1 = stats.norm.ppf(alpha/2, mu0, s)
			k2 = stats.norm.ppf(1-alpha/2, mu0, s)
			x = np.linspace(k1-4*s, k2+4*s, 1000)
		case _:
			err = "The 'riktning' parameters must be one of " + \
				"'!=', '<' or '>'; not " + format(riktning,'s')
			raise ValueError(err)
	#end match case
	h = 1 - (stats.norm.cdf(k2, x, s) - stats.norm.cdf(k1, x, s))
	
	#if mu_sant given compute power at that point
	if not (mu_sant is None):
		f = 1 - (stats.norm.cdf(k2, mu_sant, s) - stats.norm.cdf(k1, mu_sant, s))
	else:
		f = None
	#if not (mu_sant is None):

	#create subfigures
	fig,axs = plt.subplots(2-(mu_sant is None), 1, constrained_layout=True)
	#and pick up axis for basix plot
	ax_h = axs if mu_sant is None else axs[0]

	#plot h(x)
	ax_h.plot(x, h, color='b')
	#add line illustrating power at mu_sant
	ax_h.plot([mu0,mu0,x[0]], [0,alpha,alpha], color='k', linestyle=':')
	#create title
	title = r'h($\mu$) = P(förkasta $H_0$)'
	if not (mu_sant is None):
		title = title + '; h(' + format(mu_sant,'2.1f') + ') = ' + \
			format(f, '2.2f')
		#add line illustrating power at mu_sant
		ax_h.plot([mu_sant,mu_sant,x[0]], [0,f,f], color='r', linestyle=':')

	#set title
	ax_h.set_title(title)
	ax_h.set_xlabel(r'$\mu$')
	#set xlim and ylim
	ax_h.set_xlim( (min(x),max(x)) )
	ax_h.set_ylim( (-0.1, 1.1) )

	#add text annotation - position
	match riktning:
		case "<":
			x_text = max(x)-2*s
		case ">":
			x_text = min(x)+s
		case _: #the != case
			x_text = mu0-0.5*s
	#and text, y-max is 1.1
	y_text = 1
	ax_h.text(x_text, 0.9*y_text, r'H$_0$: $\mu$ = ' + format(mu0,'4.1f'))
	ax_h.text(x_text, 0.8*y_text, r'H$_1$: $\mu$ ' + riktning + ' ' + \
		   format(mu0,'4.1f'))
	ax_h.text(x_text, 0.7*y_text, 'n = ' + format(n,'4d'))
	ax_h.text(x_text, 0.6*y_text, r'$\sigma$ = ' + format(sigma,'4.1f'))
	ax_h.text(x_text, 0.5*y_text, r'$\alpha$ = ' + format(alpha, '4.3f'))

	#if mu_sant not given we are done
	if mu_sant is None:
		return fig
	
	#otehrwise do plots for H1 and H0 comparisson
	#compute maximum of density function
	f_0 = stats.norm.pdf(0,0,s)
		
	#compute density functions
	y = stats.norm.pdf(x, mu0, s)
	y_H1 = stats.norm.pdf(x, mu_sant, s)
		
	axs[1].plot(x, y_H1, color='b')
	axs[1].plot(x, y, color='k')
	#set xlim and ylim
	axs[1].set_xlim( (min(x),max(x)) )
	axs[1].set_ylim( (0, 1.2*f_0) )

	#arean forkasta inte H0
	axs[1].fill_between(x[(k1<x) & (x<k2)], y_H1[(k1<x) & (x<k2)], color='b')
	#arean forkasta inte H1
	axs[1].fill_between(x[x<k1], y[x<k1], color='r')
	axs[1].fill_between(x[x>k2], y[x>k2], color='r')
	
	#final guide lines
	axs[1].plot([mu0,mu0], [0,f_0], color='k', linestyle='--')
	axs[1].plot([mu_sant,mu_sant], [0,f_0], color='k', linestyle='--')

	#return axes
	return fig

	

