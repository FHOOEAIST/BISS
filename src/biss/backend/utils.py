import glob, os
import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib.type_check import imag
from skimage import io
from skimage.filters import gaussian
from typing import Tuple
from shutil import rmtree
from .region_grow import RegionGrowth

UINT16_MAX_ = np.iinfo(np.uint16).max

def extend_dim(indices:tuple, bound, target_size=32):
    v_min, v_max = indices
    to_add = target_size-((v_max-v_min)%target_size)
    if to_add % 2 == 0:
        v_min -= (to_add // 2)
        v_max += (to_add // 2)
    else:
        v_min -= ( (to_add // 2) + 1)
        v_max += ( to_add // 2 )

    if v_max > bound:
        diff = v_max - bound
        if diff < 0:
            raise Exception("Out of bounds at extension of image!")
        v_min -= diff
        v_max = bound
    return (v_min, v_max)

# Find bounds of image and masks according to outer bounds of brainmask
def find_bounds(brainmask,patch_shape=(32,32,32)) -> dict:
    z = np.any(brainmask, axis=(1, 2))
    r = np.any(brainmask, axis=(0, 2))
    c = np.any(brainmask, axis=(0, 1))

    # Get bounding indices and extend in order to be divisible by patch_shape
    rmin, rmax = extend_dim(np.where(r)[0][[0, -1]],brainmask.shape[1], patch_shape[1])
    cmin, cmax = extend_dim(np.where(c)[0][[0, -1]],brainmask.shape[2], patch_shape[2])
    zmin, zmax = extend_dim(np.where(z)[0][[0, -1]],brainmask.shape[0], patch_shape[0])

    return {"zmin":zmin,"zmax":zmax,"rmin":rmin,"rmax":rmax,"cmin":cmin,"cmax":cmax}

# Slice a image using given dictionary with bounds
def cut_bounds(bounds:dict, img):
    return img[bounds["zmin"]:bounds["zmax"],bounds["rmin"]:bounds["rmax"], \
            bounds["cmin"]:bounds["cmax"]]

def array3d_to_patches(arr3d_in, window_shape, step=1):
    if not isinstance(arr3d_in, np.ndarray):
        raise TypeError("`arr3d_in` must be a numpy ndarray")

    ndim = arr3d_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr3d_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr3d_in.shape`")

    arr_shape = np.array(arr3d_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr3d_in.strides)

    indexing_strides = arr3d_in[slices].strides

    win_indices_shape = (
        (np.array(arr3d_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    img_out = as_strided(arr3d_in, shape=new_shape, strides=strides)
    return img_out


def clear_folder(path):
    filelist = glob.glob(os.path.join(path, "*"))
    for f in filelist:
        os.remove(f)


def reassemble_patches(patches: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:

    if (len(patches.shape) != 6):
        raise Exception("Array must be of shape (n1,n2,n3,x,y,z)")

    i_h, i_w, i_c = img_shape
    image = np.zeros(img_shape, dtype=patches.dtype)

    n_z, n_r, n_c, p_z, p_r, p_c = patches.shape

    s_z = 0 if n_z <= 1 else (i_h - p_z) / (n_z - 1)
    s_r = 0 if n_r <= 1 else (i_w - p_r) / (n_r - 1)
    s_c = 0 if n_c <= 1 else (i_c - p_c) / (n_c - 1)

    # Step size should be same for all patches
    if (int(s_r) != s_r) or (int(s_z) != s_z) or (int(s_c) != s_c):
        raise Exception("Patch size must be the same for all patches!")

    s_r = int(s_r)
    s_z = int(s_z)
    s_c = int(s_c)

    i, j, k = 0, 0, 0

    while True:
        i_o, j_o, k_o = i * s_z, j * s_r, k * s_c

        image[i_o : i_o + p_z, j_o : j_o + p_r, k_o : k_o + p_c] = patches[i, j, k]

        if k < n_c - 1:
            k = min((k_o + p_c) // s_c, n_c - 1)
        elif j < n_r - 1 and k >= n_c - 1:
            j = min((j_o + p_r) // s_r, n_r - 1)
            k = 0
        elif i < n_z - 1 and j >= n_r - 1 and k >= n_c - 1:
            i = min((i_o + p_z) // s_z, n_z - 1)
            j = 0
            k = 0
        elif i >= n_z - 1 and j >= n_r - 1 and k >= n_c - 1:
            break # Reached image border
        else:
            raise Exception("Could not reassamble patches!")

    return image


class Preprocessor:
    def __init__(self, imagepath):
        self.__patch_history = {}
        self.__imagepath= imagepath
        self.__targetpath = None
        self.__vessels_present = self.__check_vessels()
        self.__brainmasks = {}
        self.__images = {}
        self.__check_folders()
        self.__current_target = None

    def __check_vessels(self):
        pathVessels = self.__imagepath + '/raw/vessels/'
        return os.path.exists(pathVessels) and len(glob.glob(pathVessels + '*.tif')) > 0
    

    def __check_folders(self):
        # Set target path to store patches
        self.__targetpath = self.__imagepath + '/patches/'
        if not os.path.exists(self.__targetpath):
            os.mkdir(self.__targetpath)
        if not os.path.exists(self.__targetpath + 'images/'):
            os.mkdir(self.__targetpath + 'images/')
        if self.__vessels_present and not os.path.exists(self.__targetpath + 'vessels/'):
            os.mkdir(self.__targetpath + 'vessels/')

        # Check source path for images
        for file in glob.glob(self.__imagepath+'/raw/images/*.tif'):
            img_path = file.replace("\\", "/")
            target_name = img_path.split('/')[-1].split('.tif')[0]

            mask_path = (self.__imagepath+f'/raw/brainmasks/{target_name}_brain.tif')

            if os.path.exists(mask_path):
                self.__images[target_name] = img_path
                self.__brainmasks[target_name] = mask_path
            else:
                print(f"Missing brainmask for {target_name}. Skipping image!")

           
        clear_folder(self.__targetpath + 'images/')

        if self.__vessels_present:
            clear_folder(self.__targetpath + 'vessels/')

    def __get_patches(self, img, patch_shape, step_size):
        img_patches = array3d_to_patches(img, patch_shape, step=step_size)
        self.__patch_history[self.__current_target]["patch"] = img_patches.shape
        img_patches = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
        self.__patch_history[self.__current_target]["reshape"] = img_patches.shape

        # Add single colour channel as last dimension
        return np.expand_dims(img_patches, -1)

    def store_patches(self, patch_shape=(32,32,32), step_size=32, gauss_val=0.5) -> dict:
        self.__check_folders()

        if not self.__images:
            raise Exception("No images could be found. Maybe check image path!")

        for img in self.__images.keys():
            self.__current_target = img
            print("Processing image: %s\n-----------------------------" %(img))

            self.__patch_history[img] = {}
            # Get image and brainmask from file 
            image = self.get_image(img)
            brainmask = self.get_brainmask(img)
            
            self.__patch_history[img]['orig'] = image.shape

            # Crop image using brainmask
            image *= brainmask
            
            bounds = find_bounds(brainmask)
            self.__patch_history[img]['bounds'] = bounds
            image = cut_bounds(bounds,image)

            self.__patch_history[img]['cut'] = image.shape

            # Apply gauss filter if is set > 0.0
            if (gauss_val > 0.0):
                image = gaussian(image,gauss_val)

            # Global normalization
            image = image.astype('float')
            image /= image.max()
            image *= np.iinfo(np.uint16).max
            image = image.astype(np.uint16)

            # Create patches
            img_patches = self.__get_patches(image,patch_shape,step_size)

            for i,img_patch in enumerate(img_patches):
                target_path_img = self.__targetpath + 'images/' + img + f'_{i:03}.npy'
                np.save(target_path_img, img_patch)

            # Only process vessels if present
            if self.__vessels_present:
                vessels = io.imread('%s/raw/vessels/%s_vessels.tif' %(self.__imagepath,img))

                vessels *= brainmask
                vessels = cut_bounds(bounds,vessels)

                vessel_patches = self.__get_patches(vessels,patch_shape,step_size)

                for i,vessel_patch in enumerate(vessel_patches):
                    target_path_img = self.__targetpath + 'vessels/' + img + f'_{i:03}.npy'
                    np.save(target_path_img, vessel_patch)

            print("Number patches stored: %s\n" %(len(img_patches)))


    def get_image(self, key_image):
        # Get image of given key
        if key_image in self.__images:
            return io.imread(self.__images[key_image])
        else:
            raise Exception(f"Key for image: {key_image} not found!" )

    def get_brainmask(self, key_brainmask):
        # Get brainmak of given key
        if key_brainmask in self.__brainmasks:
            return io.imread(self.__brainmasks[key_brainmask])
        else:
            raise Exception(f"Key for image: {key_brainmask} not found!" )

    @property
    def current_target(self):
        return self.__current_target
        
    @property
    def patch_history(self):
        return self.__patch_history

def save_prediction(path,pred):
    # Save image with probabilties as uint16 if no threshold given
    pred_out = pred.astype(np.uint8)
    pred_out *= np.iinfo(np.uint8).max
    io.imsave(path,pred_out,plugin='tifffile')


def post_process(pred:np.ndarray,thresh_seeds = 0.64, thresh_lower = 0.45) -> np.ndarray:
    # Get seedpoints
    seedpoints = np.argwhere( pred > (thresh_seeds*UINT16_MAX_) )
    seedpoints = seedpoints.astype(np.intc)

    # Dirty bugfix here --> pass empty int array to region grow module
    outMask = np.zeros((pred.shape), dtype=np.uint8)

    lowerThresh = int(thresh_lower*UINT16_MAX_)

    out = RegionGrowth.RegionGrow3D(pred,UINT16_MAX_,lowerThresh).apply(seedpoints, outMask, update=False)
    return np.array(out)

    #return pred > (0.43*UINT16_MAX_) # THRESHOLD INSTEAD OF REGION GROWING


def get_volume_share(pred, brainmask) -> float:
    """Get the relative proportion of blood vessels to given brain mask

    Args:
        pred (np.ndarray()): Binary segmentation of blood vessels
        brainmask (np.ndarray()): Binary mask for brain

    Returns:
        float: Relative proportion of blood vessels
    """
    return pred.sum() / brainmask.sum()