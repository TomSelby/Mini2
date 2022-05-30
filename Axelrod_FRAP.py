# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:47:46 2022

@author: tas72
"""

#%% Import modules
import matplotlib as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import curve_fit
from scipy import signal
from skimage import io
import os
from skimage import data
import imageio
import glob
#%% Import data

experiment = 'Ni(2)'# in form of POPC(1), or Ni(2) or SNAP(3) or SNAP(4)# done Ni(2)

slide = [1,2,3]
repeat = [1,2,3]
#Background
os.chdir(r'C:\Users\tselb\OneDrive - University of Cambridge\Documents\Mini 2\FRAP')
os.chdir(r'Data\Control\drive-download-20220317T085515Z-001\well9_561nm_OD2_POPC(c)_1')
filename_bg = 'well9_561nm_OD2_POPC(c)_1_MMStack_Pos0.ome.tif'
bg = io.imread(filename_bg)

os.chdir(r'C:\Users\tselb\OneDrive - University of Cambridge\Documents\Mini 2\FRAP')
os.chdir(r'Data\Control\drive-download-20220317T085515Z-001\well10_561nm_OD2_POPC(c)_1')
filename_bg2 = 'well10_561nm_OD2_POPC(c)_1_MMStack_Pos0.ome.tif'

bg2 = io.imread(filename_bg2)

background = np.mean(np.concatenate((bg,bg2))) # find the average background counts
background= background.astype(float)

#Define an ultimate df 
ultimate_df = pd.DataFrame(columns={'Filename','Diff Coeff', 'Frac Mobile'})
ultimate_df=ultimate_df.set_index('Filename')



#Each image pre, post and metadata
for i in range(len(slide)):
    for j in range(len(repeat)):
        filename = 'well'+experiment[-2]+'_slide' + str(slide[i]) + '_561nm_OD1p6_'+experiment+'-TMR_0p5mgmL_Epi_' + str(repeat[j]) # set up so that goes through each directory (filename)
        os.chdir(r'C:\Users\tselb\OneDrive - University of Cambridge\Documents\Mini 2\FRAP\Data\2022_03_17\2022_03_17_SLB_TR\plottable/'+experiment[-2]+'/'+filename) #Change the directory 
        im_post = io.imread(r'post/cropped '+filename+'_MMStack_Pos0.ome.tif')
        im_pre = io.imread(r'pre/cropped '+filename+'_MMStack_Pos0.ome.tif')
        im_post = im_post.astype(float) #Convert the entries in each array to floats
        im_pre = im_pre.astype(float) #Convert the entries in each array to floats
        print(np.shape(im_pre))
        print(np.shape(im_post))
        md = pd.read_json(filename+'_MMStack_Pos0_metadata.txt') # Import the meta data
        
        im_post = im_post-background #Correct for bg counts
        im_pre = im_pre-background #Correct for bg counts


        #%% Take the pre from the post to find the bleached spot
        spot = im_post[0]-im_pre[-1] #take the last image from pre from the first image post- only left with the spot (negative values)
        spot= gaussian_filter(spot,sigma=10) #gaussian blur
        
        
        
        #%% Set a limit and find the spot
        
        limit = np.mean(spot)*2  #set a thershold of all values a 2 over the mean value (this will be negative)
        spot_df= pd.DataFrame(data=np.empty((512, 1)),columns={'Col'},dtype=object)
        for k in range(512):
                spot_index = [idx for idx, val in enumerate(spot[k,:]) if val < limit] #Get index of values that are below the limit
                spot_df.loc[k,'Col'] = spot_index
        spot_df=spot_df[spot_df['Col'].map(lambda d: len(d)) > 0] #Remove empty lists
        
        
        
        #%% Correct for uneven illumintaion
        mean_pre = np.empty((512,512))  #create an empty array for the averaged pre image
        for k in range(len(im_pre[1])): #Iterate over all rows
            for l in range(len(im_pre[2])): #Iterate over all columns
                mean_pre[l][k] = np.mean(im_pre[:,l,k]) #Calculate the average of each pixel and set the value of the mean array
                
                
                
                
                
        for k in range(len(im_post[:,0,0])): # Divide all images after bleach by the pre bleach average
            im_post[k]= im_post[k]/mean_pre
    
    
    
        for k in range(len(im_pre[:,0,0])): # Divide all images pre bleach by the pre bleach average
            im_pre[k]= im_pre[k]/mean_pre
            
            
        #%% Find temporal fluctuations function from reference area    
        limit = np.mean(spot) #set a thershold of all values a over the mean value, spot= (first postbleach-last pre bleach)
        ## Only find the reference area once
        reference_df= pd.DataFrame(data=np.empty((512, 2)),columns={'Col', 'Ave_Col'},dtype=object) # Create an empty df
        for k in range(512):
                reference_index = [idx for idx, val in enumerate(spot[k,:]) if val > limit] #Get index of values that are above the limit(mean)
                reference_df.loc[k,'Col'] = reference_index
        reference_df=reference_df[reference_df['Col'].map(lambda d: len(d)) > 0] #Remove empty lists- should'nt be any in this case
        
        temporal_array= np.empty((1,len(im_post))) #Create an array to put the temporal change data in
        
        for k in range(len(im_post)):
            ## Calculate the average intensity at each 
            ref_array= np.empty((1,512))
            for l in range(len(reference_df['Col'])): #Iterate over all the rows
                ref_array[:,l]= np.mean(im_post[k][reference_df.index[l]]) #Calculate the mean at each row for each column
            ave_ref_int =  np.mean(ref_array)# calculate the mean at each column (all the reference pixels)
            temporal_array[:,k]= ave_ref_int
            
        #%% Fit an exponential decay to the temporal fluctuations
        x=np.array([i for i in range(len(im_post))])
        def func(x,a,b):
            return (a*(np.exp(-x/b)))
        popt, pcov = curve_fit(func, x, temporal_array[0,:], maxfev=100000000)#, p0=[13000,4000000,-1300])#, method='lm')
        perr_sd = np.sqrt(np.diag(pcov))
        a, b = popt
        #%% Correct data for temporal variations
        for k in range(len(im_post[:,0,0])): # Divide all images after bleach by function which gives the amount photofaded
            im_post[k]= im_post[k]/func(k,a,b)
            
            
            
        #%% Fit a 2D gaussian to the images
        
        def expguass(VARS,x0,y0,depth,Re):
            X,Y= VARS
            return np.exp(-depth*np.exp((-2*((X-x0)**2+(Y-y0)**2))/(Re**2))) #Define the expnonential gaussain


        df= pd.DataFrame(columns=['x0', 'y0', 'Depth', 'Re','Intensity at Re=sd','Time (s)'],index=np.arange(len(im_post))) #Create a df to put the fittted parameters in
        
        x,y = np.arange(0, 512), np.arange(0, 512) #Define the x and y grid for fitting- as a 2D function need to use meshgrid and later ravel
        X, Y = np.meshgrid(x, y)
        
        #cen_x = np.median(spot_df.loc[round((len(spot_df)/2))][0])#  find the middle column of the middle row- just for initial guess with fitting
        #cen_y = spot_df.index[round((len(spot_df))/2)] #find the middle of the rows- just for initial fitting guess
        
        xdata = np.vstack((X.ravel(), Y.ravel())) # Define xdata for fitting (really x and y data)
        for k in range(len(im_post)): # Perform fitting over however many images you need
            Z= im_post[k] #Define the Z data as your image data
            
            popt,pcov = curve_fit(expguass,xdata,Z.ravel(),p0=[250,250,0.5,100],maxfev = 100000) #Perform fit- guess 250 for the center 
            x0,y0,depth,Re= popt
            
            #perr_sd = np.sqrt(np.diag(pcov)) #Calculate the error on the fit
            df.iloc[k] = {'x0': x0, 'y0': y0,'Depth':depth,'Re':Re} #Place the values in the df
            
        df['Intensity at Re=sd'] = np.exp(-(df['Depth'].astype(float))*(np.exp(-2)))
        bleaching_frames = 300-len(im_pre)-len(im_post) # Find the number of frames which were removed when split into pre and post bleach
        df['Time (s)']= (((md.loc['ElapsedTime-ms'][1+bleaching_frames+len(im_pre):]- md.loc['ElapsedTime-ms'][1+bleaching_frames+len(im_pre)])/1000).to_numpy())[:] #Want to start the timer count at the start of im_post (1+to remove column title) 
        df.to_csv(filename+'2D gaussian.csv')
        print(df.head())
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
        Finf=a*(1-np.exp(-999999999999/b))+c #Just use a very large number for Finf
        frac_mobile = (Finf-F0)/(Fi-F0)
        print(frac_mobile)
        #Perform taylor expansion and find Diff coeff
        def texpansion(t,K,Td):
            total = float(0)
            for n in range(100): #gives the number of terms you want in your expansion
                total = total + (frac_mobile*((-K)**n)/((np.math.factorial(n))*(1+n*(1+(2*t/Td)))))  #Add up all the terms in the serise expansion
            return total + (1-frac_mobile)*F0 #return the sum

        popt,pcov = curve_fit(texpansion,df['Time (s)'],df['Intensity at Re=sd'],p0=[-1,1],maxfev=10000) #optimse for the sum of the t expansion
        
        K,Td = popt
        pixel_length = (16e-6/150)
        w= np.mean((df['Re']))*pixel_length
        D=(w**2)/(4*Td)
        print(str(D/((10**-6)**2))+'  microm2/s')
        ultimate_df=ultimate_df.append({'Filename':filename,'Diff Coeff':D, 'Frac Mobile':frac_mobile},ignore_index=True)


ultimate_df.to_csv(filename+'ultimate_df.csv') #for each experiment give an ultimate df with the calculated D and frac mob for each repeat
