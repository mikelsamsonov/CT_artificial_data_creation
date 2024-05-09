'''
Creation of artificial projection and reconstruction data for Computed Tomography setups in order to get more training data for further use of Concolutional Neural Networks in radiology. Only pre-processing of Computed Tomography data is given due to NDA. 
'''

import numpy as np
import pyCT        
import matplotlib.pyplot as plt        
import pyqtgraph as qt        
import scipy.ndimage as scnd
import pyE17 as py

# data loading, dark_field and flat_field projections are measured with less angles in order to save time and computing power

raw_projection_list_full = [py.io.h5read('/data/raw_projection/xrd_01472_00003_0%04d.h5'%i)[u'raw_data'][0,:,:] for i in range(0, 1501, 1)]
dark_field_projection_list_full = [py.io.h5read('/data/dark_field/xrd_01472_00000_000%02d.h5'%i)[u'raw_data'][0,:,:] for i in range(0, 100, 1)]
flat_field_projection_list_full = [py.io.h5read('/data/flat_field/xrd_01472_00002_000%02d.h5'%i)[u'raw_data'][0,:,:] for i in range(0, 100, 1)]

raw_projection_array_full = np.array(raw_projection_list_full)
dark_field_projection_array_full = np.array(dark_field_projection_list_full)
flat_field_projection_array_full = np.array(flat_field_projection_list_full)

raw_projections = raw_projection_array_full.astype('float32') 
dark_field_projections = dark_field_projection_array_full.astype('float32')
flat_field_projections = flat_field_projection_array_full.astype('float32') 

# looking for right region for further reconstruction
plt.figure()
plt.imshow(raw_projection_array_full[1000, 880:2160, 1500:1700])   # region in the middle 

raw_projections_size_1280_200 = raw_projections[:, 880:2160, 1500:1700]
dark_field_projections_size_1280_200 = dark_field_projections[:, 880:2160, 1500:1700]
flat_field_projections_size_1280_200 = flat_field_projections[:, 880:2160, 1500:1700]

# geometry parameters
source_2_center = 73.5    #  cm
center_2_proj = 142.0     #  cm
# voxel_size = 2e-3
magnification = (source_2_center+center_2_proj)/source_2_center 
pixel_size = 0.0150  # cm, takes into account magnification => (source_2_center+center_2_proj)/source_2_center * voxel_size
voxel_size = pixel_size / magnification      # cm 

nangles = 1501
angles = -np.linspace(0, 360, nangles, False)

# averaging images of dark_field and flat_field with size_1280_200
dark_field_projections_mean = dark_field_projections_size_1280_200.mean(0)
flat_field_projections_mean = flat_field_projections_size_1280_200.mean(0)

# averaged images of dark field with right size corresponding to raw projections size with size_1280_200
corrected_raw_projection_half_frame = np.zeros(((1501, 1280, 200)))
dark_field_with_right_size = np.zeros(((1501, 1280, 200)))
flat_field_with_right_size = np.zeros(((1501, 1280, 200)))

for i in range(0, 1501):
	dark_field_with_right_size[i, :, :] = dark_field_projections_mean
	flat_field_with_right_size[i, :, :] = flat_field_projections_mean

# correction size_1280_200
corrected_raw_projection_size_1280_200 = (raw_projections_size_1280_200 - dark_field_with_right_size) / (flat_field_with_right_size - dark_field_with_right_size)

# THIS CORRECTION FOR NANS AND DEAD PIXELS IS USED ONLY WITH BIGGER REGION, BECAUSE WITH SIZE 1280_200 THERE IS ONLY 1 NAN.

# How to find nans with half-frame 
np.isnan(corrected_raw_projection_half_frame[0]).sum()  # gives sum of true values in boolean array (of nans)
nans_number_each_projection = np.isnan(corrected_raw_projection_half_frame[0]).sum()
np.argwhere(np.isnan(corrected_raw_projection_half_frame[0]))  # gives sum coordinates of true values (of nans) in boolean array 

# Correction for dead pixel with half-frame, usually neighbour pixel appears not to be dead pixel
for i in range(1501):
	for j in range(nans_number_each_projection):
		corrected_raw_projection_half_frame[i, np_array[j][0], np_array[j][1]] = corrected_raw_projection_half_frame[i, np_array[j][0], np_array[j][1] + 1]

# Check for nans with half-frame
nans_counter = 0
for i in range(corrected_raw_projection_half_frame.shape[0]):
	nans_counter += np.isnan(corrected_raw_projection_half_frame[i]).sum()
print(nans_counter)

# log
corrected_raw_projection_size_1280_200_log = -np.log(corrected_raw_projection_size_1280_200)


# setting projector
P = pyCT.P.Cone_SpP_lin(angles = angles)

# define some geometry parameters
P.source_2_center = source_2_center
P.center_2_proj = center_2_proj 
# P.proj_area_size 
P.proj_pixel_size = pixel_size
# P.img_volume_size  
# P.img_volume_size  
P.img_voxel_size = voxel_size
P.proj_center_shift = 2
# P.angles[:, 0] = 0.5   # after that projector does not let setting

# Reconstruction
recon = P.fbp(cropped_corrected_raw_projection_size_1280_200_log.swapaxes(1,2), filter_type='ram-lak')   # swapaxes(1,2) because P.fbp() function expects such input 
qt.image(recon.swapaxes(0,1))  # visualization

'''
fbp() function is more advanced version of forwardproject() function with more geometrical configurations involved
def forwardproject(sample, angles):
    """
    Simulate data aquisition in tomography from line projections.
    Forwardproject a given input sample slice to obtain a simulated sinogram.
    """
    sh = np.shape(sample)                # calculate shape of sample
    Nproj = len(angles)                  # calculate number of projections

    # define empty sinogram container, angles along y-axis
    sinogram = np.zeros((Nproj, sh[1]))   
    # we generate sinogram image which consists of Number of angles (301) and the shape of the sample

    # loop over all projections, so we rotate the sample every time a little bit (на (360/301) градусов)
    for proj in np.arange(Nproj):
        print("Simulating:     %03i/%03i" % (proj+1, Nproj), end="\r", flush=True)
    # flush=True значит сразу записывает во время цикла    
        
        im_rot = nd.rotate(sample, angles[proj], reshape=False)
            # angles[proj] это типа угол определенный из списка (будет дальше), proj = 0,1,2,..., 
            # reshape=False because we want to keep sample shape (чтобы детектору всегда хватало пикселей точь в точь)
        sinogram[proj, :] = np.sum(im_rot, axis=0)
            # sinogram is just sum over pixels on a line for 1 angle theta = proj (логично т.к. axis=0 это в столбик, 
            # т.е. соответсвует сумме пикселей sample, которые соответствуют 1-му пикселю детектора, и так по всем пикселям)
        
    return sinogram
    
'''


# Circular mask filtering

h, w = recon.swapaxes(0,1)[100].shape

# Y, X = np.ogrid[:h, :w]

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

mask = create_circular_mask(h, w, center=None, radius=600)

masked_img = np.zeros((recon.swapaxes(0,1).shape))

for i in range(recon.swapaxes(0,1).shape[0]):
    masked_img[i,:,:] = recon.swapaxes(0,1)[i,:,:]
    masked_img[i,:,:][~mask] = 0   # keeping data only inside circular mask

qt.image(recon.swapaxes(0,1)[100])  # visualization

plt.figure()
plt.subplot(121)
plt.title('Reconstruction slice \nwithout \ncircular mask \n')
plt.tight_layout()
plt.imshow(recon.swapaxes(0,1)[100])
plt.colorbar()
plt.tight_layout()
plt.subplot(122)
plt.title('Reconstruction slice \nafter applying \ncircular mask \n')
plt.tight_layout()
plt.imshow(masked_img[100])
plt.colorbar()
plt.tight_layout()

plt.savefig('./Reconstruction slice without circular mask and reconstruction slice after applying circular mask.pdf')

# Projection
proj = P.project(masked_img.swapaxes(0,1))
qt.image(proj)  # visualization



