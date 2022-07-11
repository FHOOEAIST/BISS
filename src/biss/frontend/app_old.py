from biss.backend import BissClassifier, save_prediction, get_volume_share
from glob import glob
from os import path
import threading
import time
from IPython.display import clear_output
from IPython.display import display
#from google.colab import files
from zipfile import ZipFile as zip
from ipywidgets import Layout, Dropdown, Button, Output, FloatProgress, Label, VBox, HBox
from .viewer import ImageSliceViewer3D

# application class
class App:
  def __init__(self, basepath):
    self.__basepath = basepath
    self.bc = None     # classifier
    self.preds = None
    self.models = self.__get_models()
    self.results = None
    self.currChosen = None
    self.mainLayout = Layout(width='20%', height='60px')

    # widgets
    self.btnInit = Button(description="Initialize", button_style='primary', layout=self.mainLayout)
    self.btnPredict = Button(description="Predict", button_style='primary', layout=self.mainLayout)
    self.btnSave = Button(description="Save", button_style='primary', layout=self.mainLayout)
    
    self.outputPred = Output()
    self.outputResult = Output()

    # dropDown menus
    self.predChosen = Dropdown(
            options=self.models,
            value=None,
            description='Model:',
            disabled=False,
        )
    self.resultChosen = Dropdown(
            description='Result:',
            disabled=False,
        )

    self.__init_gui()
    
  def __get_models(self) -> list:
     return [path.basename(x) for x in glob(self.__basepath+'/models/*.h5')]
  
  def on_change(self, change):
    if change['type'] == 'change' and change['name'] == 'value':
      self.btnPredict.button_style='primary'
      self.bc.load_model('%s/models/%s' %(self.__basepath,self.predChosen.value))

  def __init_gui(self):
    self.bc = BissClassifier(self.__basepath)
    self.predChosen.observe(self.on_change)
    display(self.predChosen, self.outputPred)
    self.btnPredict.on_click(self.on_predict_clicked)
    display(self.btnPredict, self.outputPred)

  def on_predict_clicked(self, b):
    # Display the message within the output widget.
    progress = FloatProgress(value=0.0, min=0.0, max=1.0)
    finished = False
    total = 30
    def work(progress, total):
      i = 0
      while finished != True:
        progress.value = float(i+1)/total
        i += 1
        time.sleep(0.08)
        if i == total:
          i = 0
    if not self.bc:
      print("Set a model first")
      self.btnPredict.button_style='danger'
    else:
      # start new thread
      thread = threading.Thread(target=work, args=(progress,total,))
      display(progress)
      thread.start()
      print("Started prediction . . .")
      self.preds = self.bc.predict()
      if len(self.preds) > 0:
        self.btnPredict.button_style='success'
        finished = True
        progress.value = total
        progress.style.bar_color = 'green'
      else:
        self.btnPredict.button_style='danger'
        print("No prediction was made")

  # TODO: HOW TO SAVE IN BETTER MANNER??
  def on_btnSave_clicked(self, b):
    path = self.__basepath + "/train/raw/images/" + self.currChosen + "_pred.tif"
    zip_path = self.__basepath + "/train/raw/images/" + self.currChosen + "_pred.zip"
    # save in drive
    save_prediction(path, self.preds[self.currChosen])
    zip(zip_path, mode='w').write(path, self.currChosen + ".tif")
    # download zip file from drive
    #files.download(zip_path)
    print("Image successfully saved.")

  def __show_on_change(self, change):
    if change['type'] == 'change' and change['name'] == 'value':
      clear_output(wait=True)
      self.init_results(None, new_item=change['new'])

  def __init_results(self, new_item=""):
    self.results = list(self.preds.keys())
    if new_item == "":
      self.currChosen = self.results[0]
    else:
      self.currChosen = new_item
    self.resultChosen.options = self.results
    self.resultChosen.value = self.currChosen

    # Get result data
    brainmask = self.bc.get_brainmask(self.currChosen)
    image = self.bc.get_image(self.currChosen) * brainmask
    pred = self.preds[self.currChosen]

    # textBox for statistics
    items = [Label("Image:"), Label("Volume:"), Label("Shape:")]
    left_box = VBox([items[0], items[1], items[2]])
    stat_items = [Label(self.currChosen), Label(str(get_volume_share(pred,brainmask))), Label(str(pred.shape))]
    right_box = VBox([stat_items[0], stat_items[1], stat_items[2]])
    display(HBox([left_box, right_box]), self.outputResult)

    # show image
    ImageSliceViewer3D(pred, image, cmap='Greys', figsize=(20,20))


  def __show_gui(self):
    if self.preds and len(self.preds) > 0:
      self.resultChosen.observe(self.__show_on_change)
      self.btnSave.on_click(self.on_btnSave_clicked)
      display(self.btnSave, self.outputResult)
      display(self.resultChosen, self.outputResult)
    else:
      print("No results present")