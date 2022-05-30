# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:55:26 2022

@author: tas72
"""

#%% Import modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
#%% Import data
ultimate_df = pd.DataFrame(columns={'Filename','Diff Coeff', 'Frac Mobile','K', 'Td', 'Kerr','Tderr'})
ultimate_df=ultimate_df.set_index('Filename')
experiment = 'SNAP(4)'# in form of POPC(1), or Ni(2) or SNAP(3) or SNAP(4)# done Ni(2)

slide = [1,2,3]
repeat = [1,2,3]

#Define directory and slide/well
for i in range(len(slide)):
    for j in range(len(repeat)):
        filename = 'well'+experiment[-2]+'_slide' + str(slide[i]) + '_561nm_OD1p6_'+experiment+'-TMR_0p5mgmL_Epi_' + str(repeat[j]) # set up so that goes through each directory (filename)
        os.chdir(r'C:\Users\tselb\OneDrive - University of Cambridge\Documents\Mini 2\FRAP\Data\2022_03_17\2022_03_17_SLB_TR\plottable/'+experiment[-2]+'/'+filename) #Change the directory
        df=pd.read_csv(filename+'2D gaussian.csv')
        #%% Make a model for the recovery to get immobile fraction and plot recovery
        
        
        def recovery(x,a,b,c):
            return a*(1-np.exp(-x/b))+c
        df = df.drop(df.index[df['Intensity at Re=sd'] == np.inf].tolist(),axis=0) # Need to remove data that contains 'inf' from bad fitting
        df = df.drop(df.index[df['Intensity at Re=sd'] == np.nan].tolist(),axis=0) #similarly get rid of fitting that contains NaNs
        time=df['Time (s)'].to_numpy(dtype=float)
        Int =df['Intensity at Re=sd'].to_numpy(dtype=float)
        popt, pcov = curve_fit(recovery, time, Int, maxfev=10000, p0=[0.08,4,0.7])
        a, b, c = popt
        
        Fi= 1 # Intensity at the start before bleaching~ 1 as normalised
        F0= min(df['Intensity at Re=sd'])
        Finf=a*(1-np.exp(-9999999999999999999/b))+c #Just use a very large number for Finf
        frac_mobile = (Finf-F0)/(Fi-F0)
        print(frac_mobile)
        
        
        #Perform taylor expansion and find Diff coeff
        def texpansion(t,K,Td,Re=df['Re'],Rn=np.min(df['Re'][:20])):
            total = float(0)
            for n in range(100): #gives the number of terms you want in your expansion
                total = total + (((-K)**n)/((np.math.factorial(n))*(1+(n*(1+(2*t/Td))))))  #Add up all the terms in the serise expansion
            return total*frac_mobile + (1-frac_mobile)*F0  #return the sum
       
        
       
        
       
        def texpansion_w_change(t,K,D,Re=df['Re'],Rn=np.min(df['Re'][:20])):
            total = float(0)
            for n in range(100): #gives the number of terms you want in your expansion
                total = total + ((((-K)**n)*(Re**2))/((np.math.factorial(n))*((Re**2)+(n*(8*D*t+(Rn**2))))))  #Add up all the terms in the serise expansion
            return total*frac_mobile + (1-frac_mobile)*F0  #return the sum
        
            
            
        # popt,pcov = curve_fit(texpansion_w_change,df['Time (s)'],df['Intensity at Re=sd'],p0=[0.3,2],maxfev=10000) #optimse for the sum of the t expansion    
        # K, D = popt
        
        
        popt,pcov = curve_fit(texpansion,df['Time (s)'],df['Intensity at Re=sd'],p0=[0.3,2],maxfev=10000) #optimse for the sum of the t expansion
        K, Td = popt
        perr = np.sqrt(np.diag(pcov))
        print(K,Td)
        print(perr)
        
        # Calculate the residuals
        res = df['Intensity at Re=sd'] - texpansion(df['Time (s)'], *popt)
        
        
        fig, ax = plt.subplots(1,2)
        ax[0].plot(df['Time (s)'],df['Intensity at Re=sd'],label='Data')
        ax[0].plot(df['Time (s)'], texpansion(df['Time (s)'],*popt),alpha =0.5,linestyle='dashdot',color='red', label= 'Model')
        ax[0].legend()
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Intensity')
        ax[1].plot(df['Time (s)'],res)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()
        
        #plt.savefig(filename + '.png',dpi=600)
        
        
        
        
        pixel_length = (16/150) #in um
        w= (min(df['Re'][0:20]))*pixel_length
        D=(w**2)/(4*Td)
        #print(str(D)+'  microm2/s',K)
        ultimate_df=ultimate_df.append({'Filename':filename,'Diff Coeff':D, 'Frac Mobile':frac_mobile,'K':K, 'Td':Td, 'Kerr':perr[0],'Tderr':perr[1]},ignore_index=True)
        

ultimate_df.to_csv(filename+'ultimate_df.csv') #for each experiment give an ultimate df with the calculated D and frac mob for each repeat