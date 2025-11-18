# -*- coding: utf-8 -*-
#%% packages used by this file
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#%% funktionen skattningar
def skattningar(mu=0, sigma=1, n=(10,100), alternativ='muskatt'):
	"""
	skattningar Illustrerar mu och sigma^2-skattning samt konfidensintervall
	
	Ritar histogram for mu- och sigma^2-skattning eller illustrerar 
	konfidensintervall for mu. Skattningarna baseras på n[0] respektive n[1]
	observationer fran en normalfördelning, N(mu, sigma).

	Parameters
	----------
	mu : real number, optional
		Väntevärde i normalfördelningen. The default is 0.
	sigma : positive number, optional
		Standardavvikelse i normalfördelningen. The default is 1.
	n : Two positive integers in tuple or array, optional
		Antal observationer att simulera i de två stickproven. Gör det möjligt 
		att jämföra hur skattningarna beter sig med olika antal observationer. 
		The default is (10,100).
	alternativ : string, optional
		Vad som skall illustrerar:
		'muskatt': Histogram for skattningar av mu
		'sigmaskatt': Histogram for skattningar av sigma^2
		'konfint': Illustrerar konfidensintervall for mu.
		The default is 'muskatt'.

	Returns
	-------
    Figure object of matplotlib.figure containing plots of the densities.
    
    Johan Lindström
	"""
	#convert incomming tupplet to array
	n = np.array(n)
	#Antal simuleringar som gors
	n_sim = 1000
	#simulera tva sample
	x = stats.norm.rvs(mu, sigma, n[0]*n_sim).reshape((n_sim,n[0]))
	y = stats.norm.rvs(mu, sigma, n[1]*n_sim).reshape((n_sim,n[1]))
	
	#%% illustrerar skattningar av mu med olika n
	if alternativ=='muskatt':
		#mu estimates
		mu_est = np.column_stack( (x.mean(axis=1), y.mean(axis=1)) )
		#intervallens bredd, for att satta axlar.
		width = 3*sigma / np.sqrt(min(n))
		#subplots and the
		fig,axs = plt.subplots(2, 1, constrained_layout=True)
		for i in [0,1]:
			sns.histplot(x=mu_est[:,i], ax=axs[i], stat='density')
			axs[i].set_title('Skattning av mu, n = ' + format(n[i],'d') + 
					' observationer')
			axs[i].axvline(x=mu, color='r')
			axs[i].set_xlim(mu-width, mu+width)
		

	#%% illustrerar skattningar av s2 med olika n
	elif alternativ=='sigmaskatt':
		#sigma estimates
		s2_est = np.column_stack( (np.var(x,axis=1), np.var(y,axis=1)) )
		#intervallens bredd, for att satta axlar.
		width = max( stats.chi2.ppf(0.9995, n-1)/(n-1) )
		#subplots and the
		fig,axs = plt.subplots(2, 1, constrained_layout=True)
		for i in [0,1]:
			sns.histplot(x=s2_est[:,i], ax=axs[i], stat='density')
			axs[i].set_title('Skattning av s2, n = ' + format(n[i],'d') + 
					' observationer')
			axs[i].axvline(x=sigma**2, color='r')
			axs[i].set_xlim(0, width)

	#%%SIMULERING AV KONFIDENSINTERVALL
	elif alternativ=='konfint':
		# berakna de 1000 konfidens intervallen baserat på känt sigma
		#intervall bredd
		w = stats.norm.ppf(0.975)*sigma/np.sqrt(n)
		#mu estimates for the first 100 samples
		mu_est = np.column_stack( (x.mean(axis=1), 
							 y.mean(axis=1)) )
		#intervalls
		CI_x = np.column_stack( (mu_est[:,0]-w[0],mu_est[:,0]+w[0]) )
		CI_y = np.column_stack( (mu_est[:,1]-w[1],mu_est[:,1]+w[1]) )
		
		#testa vilka intervall som tacker mu (genom att tecken-testa gränserna)
		I_x = np.sign(CI_x-mu).prod(axis=1) == 1
		I_y = np.sign(CI_y-mu).prod(axis=1) == 1
		#beräkna andel som missar
		p = [I_x.mean(), I_y.mean()]
		#konvertera indikatorn till färg (true='r', false='b')
		I_x = ['r' if i else 'b' for i in I_x]
		I_y = ['r' if i else 'b' for i in I_y]
		
		#antal att plotta
		n_plot = 100
		#computer intervall widths for plotting
		width = 1.2*np.max( [np.abs(CI_x[0:n_plot,:]-mu).max(), 
					   np.abs(CI_y[0:n_plot,:]-mu).max()] )
		#plot
		fig,axs = plt.subplots(1, 2, constrained_layout=True)
		for i in [0,1]:
			axs[i].axvline(mu, color='k', linestyle='--')
			axs[i].set_xlim(mu-width, mu+width)
			axs[i].set_title('Intervall för mu, n = ' + format(n[i],'d'))
			axs[i].set_xlabel('Andel av 1000 som missar: ' + 
				  format(p[i],'.3f'))
		axs[0].hlines(y=np.arange(n_plot), xmin=CI_x[0:n_plot,0], 
				xmax=CI_x[0:n_plot,1], colors=I_x[0:n_plot])
		axs[1].hlines(y=np.arange(n_plot), xmin=CI_y[0:n_plot,0], 
				xmax=CI_y[0:n_plot,1], colors=I_y[0:n_plot])
	else:
		err = "The 'alternativ' parameters must be one of " + \
			"'muskatt', 'sigmaskatt' or 'konfint'; not " + \
			format(alternativ,'s')
		raise ValueError(err)
    ##return
	return fig
