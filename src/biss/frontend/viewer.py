from ipywidgets import interact, IntSlider, RadioButtons
import numpy as np
import matplotlib.pyplot as plt

class ImageSliceViewer3D:
  def __init__(self, image, pred, figsize=(15,15), cmap='Greys'):
      self.pred_orig = pred
      self.image_orig = image
      self.pred = pred
      self.image = image
      self.figsize = figsize
      self.cmap = cmap
      self.v = [np.min(image), np.max(image)]
      self.v2 = [np.min(pred), np.max(pred)]
      
      # Call to select slice plane
      interact(self.view_selection, view=RadioButtons(
          options=["top", "front", "side"], value="top", 
          description='Slice plane selection:', disabled=False,
          style={'description_width': 'initial'}))
  
  def view_selection(self, view):
      # Transpose the image to orient according to the slice plane selection
      orient = {"front": [0,2,1], "top":[1,2,0], "side":[2,0,1], }
      self.image = np.transpose(self.image_orig, orient[view])
      self.pred = np.transpose(self.pred_orig, orient[view])
      maxZ = self.pred.shape[2] - 1
      slider_slice = IntSlider(min=0, max=maxZ, step=1, continuous_update=False, description='Image Slice:')

      # Call to view a slice within the selected slice plane
      interact(self.plot_slice, z=slider_slice)
        
  def plot_slice(self, z):
      # Plot slice for the given plane and slice
      self.fig, self.axs = plt.subplots(1, 2, figsize=self.figsize)
      self.axs[0].imshow(self.image[:,:,z], cmap=plt.get_cmap(self.cmap), 
          vmin=self.v[0], vmax=self.v[1])
      self.axs[1].imshow(self.pred[:,:,z], cmap=plt.get_cmap(self.cmap), 
          vmin=self.v2[0], vmax=self.v2[1])
      self.axs[0].set_title("Vessels predicted")
      self.axs[1].set_title("Image original")