from glob import glob
#from modules.jet_calculus import read_fits
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import datetime as dt
import os
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.table import Table
from operator import itemgetter
import scipy.stats as stats
from astropy.cosmology import FlatLambdaCDM 
from scipy import optimize
from sklearn.linear_model import LinearRegression
import math
import astropy.units as u
import astropy.constants as const
import matplotlib.ticker as ticker



#-------------------
#-------------------
#---------------------------------

def derive_tb(S,major,f,ratio=1,**kwargs):
#	'''
#	Derives brightness temperature (see Kovalev et al. 2005.
#	S			: Flux density of component/Region [Jy]
#	f			: Observing Frequency [GHz]
#	major : Major axis ofcomponent/region[mas]
#	minor : Minor axis of component/region[mas]
#	'''
    
    z=0.902
    Const = 2*np.log(2)*np.square(const.c)*1e-26/(math.pi*const.k_B*np.square(np.pi/180/3.6e6))
    tb = Const*S*(1+z)/(major**2*ratio*np.square(f*1e9))
    if 'Serr' in kwargs.keys():
        Serr = kwargs.get('Serr')
        if 'majorerr' in kwargs.keys():
            majorerr = kwargs.get('majorerr')
        else:
            raise Exception('Please give a value for majorerr as well as Serr to derive error for tb.\n')
        dtb = Const*(1+z)/(ratio*np.square(f*1e9))*np.sqrt(np.square(Serr/np.square(major))+np.square(2*S*majorerr/major**3))
        #print('derived error as well.\n')
        return (tb.value,dtb.value)
    else:
        return tb.value


#### Definition of function to read data from .fits files
#### This returns the properties of components (comps), the properties of the observation (header), and the image data
#### useful for generating the map of the clear map

def read_fits(file):
	with fits.open(file) as hdulist:
		comps   = hdulist[1].data
		header  = hdulist[0].header
		img     = hdulist[0].data
		img	= img.reshape(img.shape[2],img.shape[3])
		#img			= img[0][0]
	return comps,header,img

redshift=0.902


def mastopc(z):
	cosmo=FlatLambdaCDM(H0=70.5,Om0=0.27)
	D = cosmo.angular_diameter_distance(z)
	return (D*np.pi/180/3.6e6).to(u.parsec)

mas_to_pc = mastopc(0.902).value
mas_to_rad=4.8481368e-9
mas_to_meter=mas_to_pc*30856775812799588

### Move into the folder where the fits files are stored

os.chdir("/Users/francescotravaglini/Desktop/MasterThesis/NRAO530/dynamics/43-GHz/model/fits")

#### Array containing as objects the names of the many .fits files for the study (in my case 71 different epochs, thus 71 files)

fits_file=glob('*.fits')

## sort them and put them in alphabetical order, useful if the files are named in a way that this puts them in chronological order:
## i.e. 2013_04_6comp.fits - 2013_09_7comp.fits - 2014_05_ .........
 
fits_file.sort()

# Generate empty lists that I'll then fill up with for cycles, so that depending on the dataset I don't have to change 
# every size of the arrays I need, but the change accondingly

mod_comp, mod_header, mod_img = [],[],[]
epochs=[]

name_file = [] #np.array(len(epochs))
for file in glob('*.fits'):
    name_file.append(file)

name_file = sorted(name_file)

### Import the data for each epoch iterating the read_fits function on all the .fits files in 'fits_file'

for i in name_file:
    comp,header,img = read_fits(i)
    mod_comp.append(comp)
    mod_header.append(header)
    mod_img.append(img)
    epochs.append(i)


### From the header here I obtain the date for each observation

obs_epoch=[]  
for i in range (0,len(name_file)):
    obs_epoch.append(mod_header[i]['DATE-OBS'])

sorted_comp=mod_comp   ### this list is sorted considering the fluxes of the models components, 

                       ### from brightest to faintest, it works for every other 

                       ### variable, cause when sorting its rearranging the whole components



### Here I sort over the flux because from the models obtain for each epoch I see that, 
### except in the epochs indicated in the if condition, all the components are labeled in
### a way that makes the first component of the model the core component, and then I know that the component 
### labeled 1 in each epoch (position 0 in the array) is the core feature

print('Sorting over the flux intensity')
for i in range (0,np.size(name_file)):
    if (i==2) or (i==9) or (i==29) or (i==36) or (i==50):
        sorted_comp[i]=mod_comp[i]
    else:
        sorted_comp[i]=sorted(mod_comp[i], key=itemgetter(0), reverse=True)   
   
#### From here on, I extrapolate the properties of each components and put them into specifi arrays: flux, size, RA, DEC...


flux=[] 

for i in range (0,np.size(name_file)):
    fluxes_epoch=[]
    for j in range (0,len(sorted_comp[i])):
        fluxes_epoch.append(sorted_comp[i][j][0])
    flux.append(fluxes_epoch)

n=3

for i in range (0,np.size(flux)):
    for j in flux[i]:  # range (0,np.size(flux[i])):
        if (np.size(flux[i])>n) :
            	n=np.size(flux[i])
        else :
            n=n    #### with this condition I just define the number 'n' to be the highest number of components over all the epochs
            
### The previous definition of 'n' allows me to define a 2D array with dimensions (nr. of epochs; nr. of components)
### with nr. of components being the highest possible over the whole dataset, since arrays must have defined dimensions
### Defining everything, so every flux for every epoch, 0 and then changing just the one where there is actually data from the
### model fitting

S=np.zeros((n,np.size(flux)))            ## Here I create the matrix in which I will write the values, and initially
                                         ## they are all zeroes

err_S=np.zeros((n,np.size(flux))) 

for i in range (0,np.size(flux)):
    for j in range (0,np.size(flux[i])):
        S[j][i]=flux[i][j]
        err_S[j][i]=flux[i][j]*0.05

S_cut = S

### I change all the zeroes into NaN, so that they won't show later in the plots

S_cut[S_cut==0] = np.nan

Serr = err_S

Serr[Serr==0] = np.nan

total_flux_epoch = []
for i in range (0,len(flux)):
	total_flux_epoch.append(np.sum(flux[i]))
max_flux = np.max(S_cut)


delta_x=[]
delta_y=[]

for i in range (0,np.size(name_file)):
	delta_x_epoch=[]
	delta_y_epoch=[]
	for j in range (0,len(sorted_comp[i])):
		delta_y_epoch.append(sorted_comp[i][j][2])
		delta_x_epoch.append(sorted_comp[i][j][1])
	delta_x.append(delta_x_epoch)
	delta_y.append(delta_y_epoch)


dim_1= np.shape(S)[0]
dim_2= np.shape(S)[1]

### dim_1 and dim_2 are the two dimensions of the flux array (S), which are the values I'll use to generate all 
### the other 2D arrays for dimensions and position of the components, which I'll fill up with NaN

Delta_x_centered=np.empty((dim_1,dim_2))
Delta_y_centered=np.empty((dim_1,dim_2))

Delta_x_centered[:] = np.NaN
Delta_y_centered[:] = np.NaN

dist_y_pc = np.empty((dim_1,dim_2))
dist_x_pc = np.empty((dim_1,dim_2))

dist_x_mas = np.empty((dim_1,dim_2))
dist_y_mas = np.empty((dim_1,dim_2))

dist_y_pc[:] = np.NaN
dist_x_pc[:] = np.NaN

dist_x_mas[:] = np.NaN
dist_y_mas[:] = np.NaN


for i in range (0,np.size(delta_x)):
    for j in range (0,np.size(delta_x[i])):
        Delta_x_centered[j][i] = delta_x[i][j] * 3600000.1486299 * mas_to_pc
        Delta_y_centered[j][i] = delta_y[i][j] * 3600000.1486299 * mas_to_pc


### Here I change all the RA (delta_x) and DEC (delta_y) to center every epoch around the core feature as identified
### in each epoch. This in fact is an offset that can change epoch to epoch.

for i in range (0,len(dist_x_pc)):
    offset_y = Delta_y_centered[0]
    offset_x = Delta_x_centered[0]
    dist_x_pc[i] = np.subtract(Delta_x_centered[i], offset_x)
    dist_y_pc[i] = np.subtract(Delta_y_centered[i], offset_y)
    dist_y_mas[i] = dist_y_pc[i] / mas_to_pc
    dist_x_mas[i] = dist_x_pc[i] / mas_to_pc


Dist_tot_pc = np.empty((dim_1,dim_2))

Dist_tot_pc[:] = np.NaN

for i in range (0,len(Dist_tot_pc)):
    Dist_tot_pc[i]=np.sqrt(np.power(dist_y_pc[i],2) + np.power(dist_x_pc[i],2))

Theta=np.empty((dim_1,dim_2))

Theta[:] = np.NaN

for i in range (0,len(Theta)):
    Theta[i] = np.arctan2(dist_y_pc[i],(-dist_x_pc[i]))*57.296

rms_beam=[]

for i in range (0,len(mod_header)):
	rms_beam.append(mod_header[i]['NOISE'])

radius=[]

for i in range (0,np.size(name_file)):
    radius_epoch=[]
    for j in range (0,len(sorted_comp[i])):
        radius_epoch.append(sorted_comp[i][j][3])
    radius.append(radius_epoch)

R=np.empty((dim_1,dim_2))

R[:] = np.NaN

for i in range (0,np.size(radius)):
    for j in range (0,np.size(radius[i])):
    	R[j][i]=(radius[i][j]*3600000.1486299)  # here the radius becomes written in mas



print('Use mas or pc? 0 for mas, 1 for pc')  ### Little condition useful for the production of the plots
value=int(input())                           ### where 'distance_unit' can be used in the label of the axis
if value==0:
    Radius=R
    distance_unit='mas'
elif value==1:
    Radius = R*mas_to_pc
    distance_unit='pc'


### Obtaining the beam dimension for each epoch from the header
    
    
beam_maj=[]
beam_min=[]

for i in range (0,len(mod_header)):
	beam_maj.append(mod_header[i]['BMAJ'])
	beam_min.append(mod_header[i]['BMIN'])

beam_A=[]


for i in range(0,len(beam_maj)):
	area=beam_maj[i]*beam_min[i]*(3600000.1486299**2)
	beam_A.append(area)

sigma_rms=[]

for i in range (0, len(rms_beam)):
    sigma_rms.append(rms_beam[i]*beam_A[i])

SNR=np.empty((dim_1,dim_2))

SNR[:] = np.NaN


for i in range (0,len(S)):
    for j in range (0,len(S[i])):
        SNR[i][j]=S[i][j]/sigma_rms[i]
   
sigma_d=np.empty((dim_1,dim_2))

sigma_d[:] = np.NaN

for i in range (0,len(S)):
    for j in range (0,len(S[i])):
        sigma_d[i][j]=Radius[i][j]/(SNR[i][j])


### Brightness temperature needs to be calculated, and for the 2D array
### I use the same approach, creating a NaN array from the dimensions of the flux array, and then filling each spot of the
### matrix (2D array) with the brightness temperature calculated


T_b=np.empty((dim_1,dim_2))
dT_b=np.empty((dim_1,dim_2))
radius_comp=[]
for i in range (0,len(beam_maj)):
    radius_comp.append(beam_maj[i]*3600000.1486299/5)  ### This value, that is 1/5 of the beam, 
                                                       ### is a condition for the computation of the T_b
T_b[:] = np.NaN
dT_b[:] = np.NaN
counter=0

for i in range(0,len(S)):
    for j in range (0,len(S[i])):
        if (R[i][j] < radius_comp[j]):      ### I impose the minimum size of the component to be 1/5 of the beam 
                                            ### as the limit for which the brightness temperature is obtainable
                                            ### If the size of the component is larger, I use the 1/5 of the beam
            Temp_and_err= derive_tb(S_cut[i][j],radius_comp[j],Serr=Serr[i][j],majorerr=sigma_d[i][j])
            T_b[i][j] = Temp_and_err[0] 
            dT_b[i][j] = Temp_and_err[1] 
            print('frozen value of radius '+str(counter))
            counter=counter+1
        else:
            Temp_and_err= derive_tb(S_cut[i][j],R[i][j],Serr=Serr[i][j],majorerr=sigma_d[i][j])
            T_b[i][j] = Temp_and_err[0] 
            dT_b[i][j] = Temp_and_err[1]


### Here I change the format of the observation dates
            
real_time=[]

for i in range (0,len(obs_epoch)):
    real_time.append(dt.datetime.strptime(obs_epoch[i], '%Y-%m-%d'))

dates = mplt.dates.date2num(real_time)

myFmt = mdates.DateFormatter('%Y')





##### HERE I NOW MANUALLY CHANGE THE ORDER OF THE COMPONENTS IN THE MODEL
##### AND FOR DOING SO I EXTRPOLATE THE INDICES FROM THE PLOT AND THEN
##### CHANGE FOR EVERY ARRAY  (flux, Delta_y_centered, Delta_x_centered, radius, T_B)

##################################################################################
################################################################
################
################  HERE I CREATE THE TABLE IN THE ASE OF 43 GHz


print('How many components?')

number_of_comp=int(input())

### The problem here is that I cannot change the size of arrays afterwards, so I have to know how many components 
### I have in total from the start (I used to go watch the names of the components and use the highest number, in this case 
### components are 39)

### Of these just 8 are identified, but I must use different components for the unidentified one cause there are 
### many unidentified components in single epochs, and due to how I constructed arrays here, I couldn't 
### put more than one data point per epoch for each component.

comp_no=['core']   

### Here I define the arrays that will be filled with data later. They are all the same lenght and all will 
### be firstly filled by NaN values

data={}
for i in range (0,number_of_comp+1):
    data['comp_{}'.format(i)]={'names':('flux','dist_x','dist_y','dist_tot','Theta','radius','brightness_T','err_T','err_pos','err_flux'),
            'formats':('f8','f8','f8','f8','f8','f8','f8','f8','f8','f8')}
    data['comp_{}'.format(i)]['flux']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['dist_x']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['dist_y']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['dist_tot']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['radius']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['brightness_T']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['Theta']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['err_T']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['err_pos']=np.empty(len(S_cut[0]))
    data['comp_{}'.format(i)]['err_flux']=np.empty(len(S_cut[0]))
    comp_no.append(i)
    #data.values[:] = np.NaN
core_comp = {'names':('flux','dist_x','dist_y','tot_dist','radius','brightness_T','err_pos','err_flux'),'formats':('f8','f8','f8','f8','f8','f8','f8','f8')}

for i,ind in enumerate (data):
    data[ind]['flux'][:]=np.NaN
    data[ind]['dist_x'][:]=np.NaN
    data[ind]['dist_y'][:]=np.NaN
    data[ind]['dist_tot'][:]=np.NaN
    data[ind]['radius'][:]=np.NaN
    data[ind]['brightness_T'][:]=np.NaN
    data[ind]['Theta'][:]=np.NaN
    data[ind]['err_pos'][:]=np.NaN
    data[ind]['err_flux'][:]=np.NaN
    data[ind]['err_T'][:]=np.NaN


#### Here I define the iteration over all the variables, so that I have to write how to fill the arrays just one time
#### and not for every property of the components. 

for i, variable in enumerate (data['comp_0']['names']):
    array = np.empty((len(S_cut),len(S_cut[0])))
    if variable=='flux':
        array = S_cut
    elif variable=='dist_y':
        if value==1:
            array = dist_y_pc
        elif value==0:
            array = dist_y_mas
    elif variable=='dist_x':
        if value==1:
            array = dist_x_pc
        elif value==0:
            array = dist_x_mas
    elif variable=='brightness_T':
        array = T_b
    elif variable=='radius':
        array = Radius
    elif variable=='err_T':
        array = dT_b

#####  Let's take  'data['comp_2'][variable][10]= array[1][10]'  as an example of how this works
#####  Left of the equal sign: the first square parenthesis is the component number, while [variable] indicates 
#####  the iteration over the different
#####  properties that the script will do. [10] indicates the this is the 10-th position of the array, describing the specific
#####  property of the component in the 10-th epoch.
#####
#####  Right of the sign: array changes with the for cycle, and indicates which variable I'm filling the data-array with.
#####  [1] indicates the component number as in the .fits file from the model while [10] is the epoch, since the flux, distance,
#####  size, etc.... are into 2D arrays where one of the dimension is the nr. of epochs
        
        
        
    data['comp_2'][variable][0]= array[1][0]
    data['comp_2'][variable][1]= array[1][1]
    data['comp_2'][variable][2]= array[1][2]
    data['comp_2'][variable][3]= array[1][3]
    data['comp_2'][variable][4]= array[1][4]
    data['comp_2'][variable][5]= array[1][5]
    data['comp_2'][variable][6]= array[2][6]
    data['comp_2'][variable][7]= array[1][7]
    data['comp_2'][variable][8]= array[1][8]
    data['comp_2'][variable][9]= array[1][9]
    data['comp_2'][variable][10]= array[1][10]
    data['comp_2'][variable][11]= array[1][11]
    data['comp_2'][variable][12]= array[1][12]
    data['comp_2'][variable][13]= array[2][13]
    data['comp_2'][variable][14]= array[2][14]
    data['comp_2'][variable][15]= array[2][15]
    data['comp_2'][variable][16]= array[2][16]
    data['comp_2'][variable][17]= array[2][17]
    data['comp_2'][variable][18]= array[2][18]
    data['comp_2'][variable][19]= array[4][19]
    data['comp_2'][variable][21]= array[3][21]
    data['comp_2'][variable][23]= array[4][23]
    data['comp_2'][variable][24]= array[3][24]
    data['comp_2'][variable][25]= array[3][25]
    data['comp_2'][variable][26]= array[4][26]
    data['comp_2'][variable][27]= array[4][27]
    data['comp_2'][variable][28]= array[6][28]
    data['comp_2'][variable][29]= array[5][29]
    data['comp_2'][variable][30]= array[4][30]
    data['comp_2'][variable][31]= array[3][31]
    data['comp_2'][variable][33]= array[5][33]
    data['comp_32'][variable][35]= array[6][35]
    data['comp_2'][variable][36]= array[5][36]
    data['comp_2'][variable][37]= array[5][37]
    data['comp_2'][variable][38]= array[5][38]
    data['comp_2'][variable][39]= array[5][39]
    data['comp_2'][variable][40]= array[6][40]
    data['comp_2'][variable][41]= array[6][41]
    data['comp_2'][variable][42]= array[6][42]
    data['comp_2'][variable][43]= array[5][43]
    data['comp_2'][variable][44]= array[6][44]
    data['comp_2'][variable][48]= array[5][48]
    data['comp_2'][variable][54]= array[7][54]
    data['comp_2'][variable][55]= array[6][55]
    data['comp_2'][variable][56]= array[5][56]
    data['comp_2'][variable][60]= array[5][60]
    data['comp_2'][variable][62]= array[4][62]
    data['comp_2'][variable][64]= array[4][64]
    
    
    data['comp_0'][variable][6]= array[1][6]
    data['comp_0'][variable][7]= array[6][7]
    data['comp_0'][variable][8]= array[3][8]
    data['comp_0'][variable][9]= array[4][9]
    data['comp_0'][variable][10]= array[4][10]
    data['comp_0'][variable][11]= array[6][11]
    data['comp_0'][variable][12]= array[4][12]
    data['comp_0'][variable][13]= array[4][13]
    data['comp_0'][variable][14]= array[4][14]
    
    
    data['comp_1'][variable][0]= array[2][0]
    data['comp_1'][variable][1]= array[2][1]
    data['comp_1'][variable][2]= array[2][2]        
    data['comp_1'][variable][3]= array[2][3]
    data['comp_1'][variable][4]= array[2][4]
    data['comp_1'][variable][5]= array[3][5]
    data['comp_1'][variable][6]= array[3][6]
    data['comp_1'][variable][7]= array[2][7]
    data['comp_1'][variable][8]= array[2][8]
    data['comp_1'][variable][9]= array[2][9]
    data['comp_1'][variable][10]= array[2][10]
    data['comp_1'][variable][11]= array[2][11]
    data['comp_1'][variable][12]= array[3][12]
    data['comp_1'][variable][13]= array[3][13]
    data['comp_1'][variable][14]= array[3][14]
    data['comp_1'][variable][15]= array[3][15]
    data['comp_1'][variable][16]= array[3][16]
    data['comp_1'][variable][17]= array[3][17]
    data['comp_1'][variable][18]= array[3][18]
    data['comp_1'][variable][19]= array[3][19]
    data['comp_1'][variable][20]= array[3][20]
    data['comp_1'][variable][21]= array[2][21]
    data['comp_1'][variable][22]= array[4][22]
    data['comp_1'][variable][23]= array[2][23]
    data['comp_1'][variable][24]= array[2][24]
    data['comp_1'][variable][25]= array[2][25]
    data['comp_1'][variable][26]= array[2][26]
    data['comp_1'][variable][27]= array[2][27]
    data['comp_1'][variable][28]= array[3][28]
    data['comp_1'][variable][29]= array[1][29]
    
    data['comp_1'][variable][32]= array[1][32]
    data['comp_1'][variable][33]= array[1][33]
    data['comp_1'][variable][34]= array[4][34]
    data['comp_8'][variable][35]= array[2][35]        
    data['comp_1'][variable][36]= array[1][36]
    
    data['comp_1'][variable][37]= array[1][37]
    data['comp_1'][variable][38]= array[2][38]
    data['comp_1'][variable][39]= array[2][39]
    data['comp_1'][variable][40]= array[2][40]
    data['comp_1'][variable][41]= array[2][41]
    data['comp_1'][variable][42]= array[2][42]
    data['comp_1'][variable][43]= array[3][43]
    data['comp_1'][variable][44]= array[1][44]
    data['comp_1'][variable][45]= array[1][45]
    data['comp_1'][variable][46]= array[2][46]
    data['comp_1'][variable][48]= array[2][48]
    data['comp_1'][variable][49]= array[2][49]
    data['comp_1'][variable][50]= array[2][50]
    data['comp_1'][variable][51]= array[2][51]
    data['comp_1'][variable][52]= array[2][52]
    data['comp_1'][variable][53]= array[2][53]
    data['comp_1'][variable][54]= array[2][54]
    data['comp_1'][variable][55]= array[2][55]
    data['comp_1'][variable][56]= array[2][56]
    data['comp_1'][variable][57]= array[2][57]
    data['comp_1'][variable][58]= array[1][58]
    data['comp_1'][variable][59]= array[1][59]
    data['comp_1'][variable][60]= array[2][60]
    data['comp_1'][variable][61]= array[2][61]
    data['comp_1'][variable][62]= array[1][62]
    data['comp_1'][variable][63]= array[1][63]
    data['comp_1'][variable][64]= array[1][64]    
    data['comp_1'][variable][65]= array[1][65]
    data['comp_1'][variable][66]= array[1][66]
    data['comp_1'][variable][67]= array[2][67]
    data['comp_1'][variable][68]= array[2][68]
    data['comp_1'][variable][69]= array[2][69]
    data['comp_1'][variable][70]= array[2][70]
 

    data['comp_18'][variable][60]= array[4][60]
    data['comp_18'][variable][63]= array[3][63]

    
    data['comp_19'][variable][4]= array[4][4]
    data['comp_19'][variable][5]= array[5][5]

    
    data['comp_13'][variable][2]= array[4][2]
    data['comp_13'][variable][4]= array[7][4]
  
    
    data['comp_3'][variable][9]= array[3][9]
    data['comp_3'][variable][10]= array[5][10]
    data['comp_3'][variable][11]= array[4][11]
    data['comp_3'][variable][12]= array[2][12]
    data['comp_3'][variable][13]= array[1][13]
    data['comp_3'][variable][14]= array[1][14]
    data['comp_3'][variable][15]= array[1][15]
    data['comp_3'][variable][16]= array[1][16]
    data['comp_3'][variable][17]= array[1][17]
    data['comp_3'][variable][18]= array[1][18]
    data['comp_3'][variable][19]= array[1][19]
    data['comp_3'][variable][20]= array[1][20]
    data['comp_3'][variable][21]= array[1][21]
    data['comp_3'][variable][22]= array[1][22]
    data['comp_3'][variable][23]= array[3][23]
    data['comp_3'][variable][26]= array[3][26]
    data['comp_3'][variable][28]= array[5][28]
    data['comp_3'][variable][29]= array[6][29]
    data['comp_3'][variable][31]= array[6][31]
    
    
    data['comp_10'][variable][31]= array[2][31]
    data['comp_10'][variable][32]= array[2][32]
    data['comp_10'][variable][33]= array[4][33]
    data['comp_10'][variable][34]= array[7][34]
 
    
    data['comp_27'][variable][54]= array[5][54]
    data['comp_27'][variable][55]= array[4][55]
    data['comp_27'][variable][56]= array[3][56]        
    
    
    data['comp_25'][variable][0]= array[3][0]
    data['comp_25'][variable][2]= array[3][2]
    data['comp_25'][variable][4]= array[6][4]


    data['comp_20'][variable][5]= array[2][5]
    data['comp_20'][variable][6]= array[4][6]
    
    
    data['comp_23'][variable][48]= array[4][48]
    data['comp_23'][variable][49]= array[4][49]


    data['comp_34'][variable][19]= array[5][19]
    data['comp_34'][variable][20]= array[7][20]
    
    data['comp_11'][variable][9]= array[6][9]
    data['comp_11'][variable][16]= array[4][16]
    data['comp_11'][variable][20]= array[5][20]
    data['comp_11'][variable][21]= array[5][21]
    
    
    data['comp_14'][variable][20]= array[4][20]
    data['comp_14'][variable][22]= array[5][22]
    
    
    data['comp_15'][variable][21]= array[4][21]
    data['comp_15'][variable][23]= array[6][23]
    data['comp_15'][variable][24]= array[5][24]


    data['comp_16'][variable][65]= array[4][65]
    data['comp_16'][variable][67]= array[8][67]

    


    data['comp_7'][variable][55]= array[3][55]
    data['comp_7'][variable][56]= array[1][56]
    data['comp_7'][variable][57]= array[1][57]
    data['comp_7'][variable][58]= array[3][58]
    data['comp_7'][variable][59]= array[5][59]
    data['comp_7'][variable][60]= array[3][60]
    data['comp_7'][variable][64]= array[3][64]
    data['comp_7'][variable][66]= array[3][66]
    data['comp_7'][variable][67]= array[3][67]
    data['comp_7'][variable][68]= array[3][68]
    data['comp_7'][variable][69]= array[3][69]
    data['comp_7'][variable][70]= array[3][70]

            
    data['comp_8'][variable][28]= array[4][28]
    data['comp_8'][variable][29]= array[3][29]
    data['comp_8'][variable][30]= array[1][30]
    data['comp_8'][variable][31]= array[1][31]
   
    
    data['comp_5'][variable][28]= array[1][28]
    data['comp_5'][variable][29]= array[2][29]
    data['comp_5'][variable][34]= array[1][34]
    data['comp_5'][variable][35]= array[1][35]
    data['comp_5'][variable][36]= array[2][36]
    data['comp_5'][variable][38]= array[1][38]
    data['comp_5'][variable][39]= array[1][39]
    data['comp_5'][variable][40]= array[1][40]
    data['comp_5'][variable][42]= array[1][42]
    data['comp_5'][variable][43]= array[1][43]
    data['comp_5'][variable][50]= array[1][50]
    data['comp_5'][variable][52]= array[1][52]

 
    data['comp_21'][variable][44]= array[4][44]
    data['comp_21'][variable][45]= array[4][45]
   
    
    data['comp_22'][variable][68]= array[5][68]
    data['comp_22'][variable][70]= array[4][70]

    
    data['comp_9'][variable][33]= array[3][33]
    data['comp_25'][variable][34]= array[2][34]
    
    
    data['comp_36'][variable][34]= array[6][34]
    data['comp_9'][variable][35]= array[4][35]
    data['comp_9'][variable][36]= array[6][36]
    
    
    data['comp_4'][variable][22]= array[2][22]
    data['comp_4'][variable][23]= array[1][23]
    data['comp_4'][variable][24]= array[1][24]
    data['comp_4'][variable][25]= array[1][25]
    data['comp_4'][variable][26]= array[1][26]
    data['comp_4'][variable][27]= array[1][27]        
    data['comp_4'][variable][28]= array[2][28]
    data['comp_4'][variable][29]= array[4][29]
    data['comp_4'][variable][30]= array[5][30]
    data['comp_4'][variable][33]= array[2][33]
    data['comp_4'][variable][34]= array[3][34]
    data['comp_4'][variable][35]= array[3][35]
    data['comp_4'][variable][36]= array[4][36]
    data['comp_4'][variable][32]= array[4][37]
    data['comp_4'][variable][33]= array[4][38]
    data['comp_4'][variable][34]= array[4][39]
    data['comp_4'][variable][35]= array[5][40]
    data['comp_4'][variable][36]= array[4][41]
    data['comp_4'][variable][42]= array[3][42]
    data['comp_4'][variable][43]= array[2][43]
    data['comp_4'][variable][44]= array[2][44]
    data['comp_4'][variable][45]= array[6][45]
    data['comp_4'][variable][46]= array[3][46]
    data['comp_4'][variable][47]= array[3][47]
    data['comp_4'][variable][48]= array[3][48]
    data['comp_4'][variable][49]= array[5][49]
    data['comp_4'][variable][50]= array[4][50]
    data['comp_4'][variable][51]= array[3][51]        
    data['comp_4'][variable][52]= array[5][52]
    data['comp_4'][variable][53]= array[3][53]
    data['comp_4'][variable][54]= array[6][54]
    data['comp_4'][variable][55]= array[5][55]
    data['comp_4'][variable][60]= array[6][60]


    data['comp_26'][variable][61]= array[3][61]
    data['comp_26'][variable][62]= array[3][62]

    
    data['comp_6'][variable][36]= array[3][36]
    data['comp_6'][variable][37]= array[3][37]
    data['comp_6'][variable][38]= array[3][38]
    data['comp_6'][variable][39]= array[3][39]
    data['comp_6'][variable][40]= array[3][40]
    data['comp_6'][variable][41]= array[1][41]
    data['comp_6'][variable][42]= array[4][42]
    data['comp_6'][variable][43]= array[4][43]
    data['comp_6'][variable][44]= array[3][44]
    data['comp_6'][variable][46]= array[1][46]
    data['comp_6'][variable][47]= array[1][47]
    data['comp_6'][variable][48]= array[1][48]
    data['comp_6'][variable][49]= array[1][49]
    data['comp_6'][variable][50]= array[3][50]
    data['comp_6'][variable][51]= array[1][51]
    data['comp_6'][variable][52]= array[3][52]
    data['comp_6'][variable][53]= array[1][53]
    data['comp_6'][variable][54]= array[1][54]
    data['comp_6'][variable][55]= array[1][55]
    data['comp_6'][variable][59]= array[2][59]        
    data['comp_6'][variable][60]= array[1][60]
    data['comp_6'][variable][61]= array[1][61]
    data['comp_6'][variable][62]= array[2][62]
    data['comp_6'][variable][63]= array[2][63]
    data['comp_6'][variable][64]= array[2][64]
    data['comp_6'][variable][65]= array[2][65]
    data['comp_6'][variable][67]= array[1][67]
    data['comp_6'][variable][68]= array[1][68]
    data['comp_6'][variable][69]= array[1][69]
            

    data['comp_28'][variable][56]= array[4][56]
    data['comp_28'][variable][57]= array[5][57]


    data['comp_29'][variable][65]= array[3][65]
    data['comp_29'][variable][66]= array[5][66]
    data['comp_29'][variable][67]= array[7][67]

    
    data['comp_30'][variable][40]= array[4][40]
    data['comp_30'][variable][41]= array[3][41]
    data['comp_30'][variable][42]= array[5][42]

            
    data['comp_24'][variable][67]= array[6][67]
    data['comp_24'][variable][68]= array[4][68]
    data['comp_24'][variable][69]= array[5][69]

    
    data['comp_17'][variable][44]= array[5][44]
    data['comp_17'][variable][45]= array[3][45]
    
    ######   THESE NEXT ARE THE UNIDENTIFIED ONES
    data['comp_32'][variable][57]= array[3][57]
    data['comp_32'][variable][34]= array[8][34]
    data['comp_2'][variable][35]= array[5][35]
    data['comp_32'][variable][37]= array[7][37]
    data['comp_8'][variable][41]= array[5][41]
    data['comp_32'][variable][48]= array[6][48]
    data['comp_32'][variable][49]= array[7][49]

            
    data['comp_33'][variable][49]= array[6][49]
    

    data['comp_34'][variable][0]= array[5][0]
    data['comp_12'][variable][3]= array[4][3]
    data['comp_34'][variable][5]= array[4][5]
    data['comp_12'][variable][6]= array[5][6]
    data['comp_34'][variable][9]= array[7][9]
    data['comp_34'][variable][10]= array[6][10]
    data['comp_34'][variable][13]= array[5][13]
    data['comp_34'][variable][14]= array[6][14]
    data['comp_34'][variable][17]= array[4][17]
    data['comp_34'][variable][19]= array[2][19]
    data['comp_34'][variable][20]= array[6][20]
    data['comp_34'][variable][26]= array[5][26]


    data['comp_35'][variable][52]= array[4][52]
    data['comp_35'][variable][54]= array[4][54]
    data['comp_35'][variable][59]= array[4][59]
    data['comp_35'][variable][67]= array[5][67]
    data['comp_35'][variable][69]= array[6][69]
    data['comp_35'][variable][53]= array[2][53]
    data['comp_35'][variable][66]= array[6][66]
    data['comp_35'][variable][70]= array[5][70]
    data['comp_35'][variable][32]= array[6][32]

    
    data['comp_36'][variable][54]= array[3][54]
    data['comp_36'][variable][55]= array[7][55]
    data['comp_36'][variable][56]= array[6][56]

    data['comp_37'][variable][7]= array[5][7]
    data['comp_37'][variable][8]= array[4][8]
    data['comp_37'][variable][23]= array[5][23]
    data['comp_37'][variable][24]= array[4][24]
    data['comp_37'][variable][25]= array[4][25]
    data['comp_37'][variable][27]= array[5][27]
    data['comp_37'][variable][28]= array[7][28]
    data['comp_1'][variable][30]= array[2][30]
    data['comp_37'][variable][31]= array[4][31]
    data['comp_37'][variable][32]= array[7][32]
    data['comp_37'][variable][33]= array[7][33]
    data['comp_37'][variable][45]= array[5][45]
    data['comp_37'][variable][46]= array[4][46]
    data['comp_37'][variable][50]= array[5][50]
    data['comp_37'][variable][53]= array[4][53]
    
    
    data['comp_38'][variable][58]= array[2][58]
    data['comp_38'][variable][59]= array[3][59]
    data['comp_38'][variable][61]= array[4][61]
    data['comp_38'][variable][63]= array[4][63]
    data['comp_38'][variable][64]= array[5][64]
    data['comp_38'][variable][65]= array[5][65]
    data['comp_38'][variable][66]= array[4][66]
    data['comp_38'][variable][67]= array[4][67]
    data['comp_38'][variable][69]= array[4][69]
    data['comp_38'][variable][70]= array[6][70]

    
    data['comp_39'][variable][0]= array[4][0]
    data['comp_39'][variable][3]= array[5][3]
    data['comp_39'][variable][7]= array[4][7]
    data['comp_39'][variable][9]= array[5][9]
    data['comp_39'][variable][14]= array[5][14]
    data['comp_39'][variable][27]= array[3][27]
    data['comp_39'][variable][30]= array[6][30]
    data['comp_39'][variable][32]= array[4][32]
    data['comp_39'][variable][35]= array[7][35]
    data['comp_39'][variable][37]= array[6][37]
    data['comp_39'][variable][45]= array[2][45]
    data['comp_39'][variable][66]= array[2][66]
    data['comp_39'][variable][70]= array[1][70]
    data['comp_39'][variable][53]= array[6][53]


    data['comp_31'][variable][4]= array[3][3]
    data['comp_31'][variable][14]= array[8][14]
    data['comp_31'][variable][27]= array[6][27]
    data['comp_31'][variable][30]= array[7][30]
    data['comp_31'][variable][32]= array[5][32]
    data['comp_31'][variable][37]= array[2][37]
    data['comp_31'][variable][45]= array[7][45]
    data['comp_31'][variable][57]= array[4][57]


### Here I define the distance from the core with the Pitagora's problem with RA-DEC coordinates

for i, comp in enumerate (data):
    for j in range (0,len(data[comp]['dist_y'])):
        if (data[comp]['dist_y'][j]>0):
            data[comp]['dist_tot'][j] = np.sqrt(np.power(data[comp]['dist_y'][j],2)+np.power(data[comp]['dist_x'][j],2))
        elif (data[comp]['dist_y'][j]<0):
            data[comp]['dist_tot'][j] = -1 * np.sqrt(np.power(data[comp]['dist_y'][j],2)+np.power(data[comp]['dist_x'][j],2))
        else:
            data[comp]['dist_tot'][j]= np.sqrt(np.power(data[comp]['dist_y'][j],2)+np.power(data[comp]['dist_x'][j],2))
        data[comp]['Theta'][j]=np.arctan2(data[comp]['dist_y'][j],(-data[comp]['dist_x'][j]))*57.296

### Here the computation of the error on the position of the components, with a lower limit 
### being 0.03 (these should be mas in my case)
        
for i, comp in enumerate (data):
    for j in range (0,len(data[comp]['dist_y'])):
        if 0.5*sigma_rms[j]*data[comp]['radius'][j]/data[comp]['flux'][j]< 0.03:
            data[comp]['err_pos'][j] = 0.03
            print('too small')
        else:
            data[comp]['err_pos'][j] = 0.5*sigma_rms[j]*data[comp]['radius'][j]/data[comp]['flux'][j]


for i, comp in enumerate (data):
    for j in range (0,len(data[comp]['dist_y'])):
        data[comp]['err_flux'][j]=data[comp]['flux'][j]*0.05

#### Here I fill the arrays for the core component, which was not fille with the others as it was not doable
#### easily with the name definition of the arrays (i.e. line~400)

core_comp['flux'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['flux'].append(S_cut[0][i])
    
core_comp['err_flux'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['err_flux'].append(S_cut[0][i]*0.05)

core_comp['dist_y'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['dist_y'].append(dist_y_pc[0][i])


core_comp['dist_x'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['dist_x'].append(dist_x_pc[0][i])


core_comp['radius'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['radius'].append(Radius[0][i])

core_comp['brightness_T'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['brightness_T'].append(T_b[0][i])
    
core_comp['err_T'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['err_T'].append(dT_b[0][i])
    
    
core_comp['Theta'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['Theta'].append(np.NaN)

#### To then operate on the data and create plots I use the astropy Tables, where I create a table of 1 'column', 
#### being this the core array, and then adding the components with simple for cycles. One table for one property.

ordered_flux = Table([core_comp['flux']], names=('core',), meta={'name':'Flux table'})

for i, item in enumerate (data):
    ordered_flux.add_column(data[item]['flux'], name=item)
 
ordered_dist_y = Table([core_comp['dist_y']], names=('core',), meta={'name':'Dist_y table'})

for i, item in enumerate (data):
    ordered_dist_y.add_column(data[item]['dist_y'], name=item)

    
ordered_dist_x = Table([core_comp['dist_x']], names=('core',), meta={'name':'Dist_y table'})

for i, item in enumerate (data):
    ordered_dist_x.add_column(data[item]['dist_x'], name=item)

core_comp['dist_tot'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['dist_tot'].append(0)
    

core_comp['err_pos'] =[]

for i in range (0,len(S_cut[0])):
    core_comp['err_pos'].append( 0.5*sigma_rms[i]*core_comp['radius'][i]/core_comp['flux'][i])


ordered_tot_dist = Table([core_comp['dist_tot']], names=('core',), meta={'name':'Dist_tot table'})

for i, item in enumerate (data):
    ordered_tot_dist.add_column(data[item]['dist_tot'], name=item)
    
    
    
ordered_Theta = Table([core_comp['Theta']], names=('core',), meta={'name':'Theta table'})

for i, item in enumerate (data):
    ordered_Theta.add_column(data[item]['Theta'], name=item)

    
ordered_Radius = Table([core_comp['radius']], names=('core',), meta={'name':'Radius table'})

for i, item in enumerate (data):
    ordered_Radius.add_column(data[item]['radius'], name=item)
    
ordered_brightness_T = Table([core_comp['brightness_T']], names=('core',), meta={'name':'Brightness Temp table'})

for i, item in enumerate (data):
    ordered_brightness_T.add_column(data[item]['brightness_T'], name=item) 


ordered_err_T = Table([core_comp['err_T']], names=('core',), meta={'name':'Error Temp table'})

for i, item in enumerate (data):
    ordered_err_T.add_column(data[item]['err_T'], name=item) 
    
    
ordered_err_pos = Table([core_comp['err_pos']], names=('core',), meta={'name':'Position error table'})

for i, item in enumerate (data):
    ordered_err_pos.add_column(data[item]['err_pos'], name=item) 


ordered_err_flux = Table([core_comp['err_flux']], names=('core',), meta={'name':'Flux error table'})

for i, item in enumerate (data):
    ordered_err_flux.add_column(data[item]['err_flux'], name=item) 
        

                        ### Just an example on the production of plots with astropy tables           
       
    fig = plt.figure('Flux of the components')
    fig.set_size_inches(15,8)
    plt.errorbar(dates,ordered_flux['core'],label= 'core', marker='+', markersize=3, c='black',linestyle='None', yerr=ordered_err_flux['core'], elinewidth=1)
    for i, item in enumerate (data):
        if i<9:
            index= int(i/3)
            plt.errorbar(dates,ordered_flux[item],label= 'comp.' + str(i+1))
    plt.plot(dates, total_flux_epoch, marker='.', linestyle=':', label='Total flux')
    plt.xlabel('epoch')
    plt.ylabel('Flux (Jy/beam)', fontsize=19)
    plt.legend(bbox_to_anchor=(0.97, 1.05),prop={'size': 11})
    plt.yscale('log')
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.savefig('/homes/ftravaglini/Documents/MasterThesis/NRAO530/dynamics/Plots_pdf/flux_43_id.pdf', dpi=250)
    
    