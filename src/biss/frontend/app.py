from biss.backend import BissClassifier, save_prediction, get_volume_share
from glob import glob
import os
from IPython.display import clear_output
from IPython.display import display
from zipfile import ZipFile as zip
from ipywidgets import Layout, Dropdown, Button, Output, FloatProgress, Label, VBox, HBox
from .viewer import ImageSliceViewer3D
from shutil import rmtree
from warnings import filterwarnings
from yaspin import yaspin

# Colab specific file handling
try:
    from google.colab import files
    from google.colab import output
    COLAB = True
    filterwarnings('ignore')
except:
    from tkinter import Tk, filedialog
    COLAB = False


class AppColab:
    def __init__(self, basepath):
        self.__basepath = basepath
        self.__bc = BissClassifier(self.__basepath)
        self.preds = None
        self.__models = self.__get_models()
        self.__modelChosen = None
        self.__predChosen = None

        self.__outputPredTag = "out_pred"
        self.__outputResultTag = "out_pred"
        self.__outputShowResultTag = "out_image"

        # Define main Layout
        self.__mainLayout = Layout(width='20%', height='60px')

        # Define global components
        self.__resultDpdwn = Dropdown( description='Result:', disabled=False)
        self.__btnPredict = Button(description="Predict", button_style='primary', layout=self.__mainLayout)
        self.__btnSave = Button(description="Save", button_style='primary', layout=self.__mainLayout)

        # Event handlers
        self.__btnPredict.on_click(self.__on_predict_clicked)
        self.__btnSave.on_click(self.__on_save_clicked)
    
        self.__init_menu_predict()

    # Get list of existing models
    def __get_models(self) -> list:
        return [os.path.basename(x) for x in glob(self.__basepath+'/models/*.h5')]

    def __on_model_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.__modelChosen = change['new']
            self.__init_menu_predict()
            if self.__bc.load_model(f'{self.__basepath}/models/{self.__modelChosen}'):
                with output.use_tags(self.__outputPredTag):
                    print(f"Loaded model {self.__modelChosen}")
            else:
                # Unset model if could no load
                self.__modelChosen = None

    def __on_predict_clicked(self, b):
      with output.use_tags(self.__outputPredTag):
        self.__init_menu_predict()
        if self.__modelChosen is None:
            print("Set a model first!")
        else:
            self.__btnPredict.disabled = True
            with yaspin(color="red") as spinner:
              self.preds = self.__bc.predict()
              #time.sleep(2)  # time consuming code

              if self.preds:
                  spinner.ok("✅ ")
              else:
                  spinner.fail("💥 ")
            
            self.__btnPredict.disabled = False

    
    def __init_menu_predict(self):
        output.clear(output_tags=self.__outputPredTag)

        with output.use_tags(self.__outputPredTag):
          modelDpdwn = Dropdown( options=self.__models, value=self.__modelChosen, description='Model:',disabled=False)
          menuPredict = VBox([modelDpdwn, self.__btnPredict])

          modelDpdwn.observe(self.__on_model_change)
          
          with output.use_tags(self.__outputPredTag):
            display(menuPredict)

    def show_results(self):
        if self.preds:
            self.__resultDpdwn.options = list(self.preds.keys())
            menuResult = VBox([self.__resultDpdwn, self.__btnSave])

            with output.use_tags(self.__outputResultTag):
              self.__predChosen = self.__resultDpdwn.options[0]

              # Display components
              display(menuResult)

              self.__updateViewer()
              self.__resultDpdwn.observe(self.__on_predictResult_change)

        else:
            with output.use_tags(self.__outputResultTag):
                print("NO RESULT TO SHOW YET")


    def __updateViewer(self):
        with output.use_tags(self.__outputShowResultTag):
              # Get result data
              brainmask = self.__bc.get_brainmask(self.__predChosen)
              image = self.__bc.get_image(self.__predChosen) * brainmask
              pred = self.preds[self.__predChosen]

              # textBox for statistics
              items = [Label("Image:"), Label("Proportion Vessels:"), Label("Shape:")]
              left_box = VBox([items[0], items[1], items[2]])
              stat_items = [Label(self.__predChosen), \
                              Label(f'{round(get_volume_share(pred,brainmask)*100,3)} %'), \
                              Label(f'{pred.shape}')]
              right_box = VBox([stat_items[0], stat_items[1], stat_items[2]])

              display(HBox([left_box, right_box]))
              ImageSliceViewer3D(pred, image, cmap='Greys', figsize=(15,15))

    def __on_predictResult_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
          output.clear(output_tags=self.__outputShowResultTag)
          self.__predChosen = change['new']
          self.__updateViewer()


    def __on_save_clicked(self, b):
        with output.use_tags(self.__outputResultTag):
            path_tmp = './tmp/'
            path_tif = path_tmp + f'{self.__predChosen}_segmented.tif'

            if not os.path.exists(path_tmp):
                os.mkdir(path_tmp)

            save_prediction(path_tif, self.preds[self.__predChosen])
            
            # download zip file from drive
            path_zip = path_tmp + f'{self.__predChosen}_segmented.zip'
            zip(path_zip, mode='w').write(path_tif, self.__predChosen + ".tif")
            files.download(path_zip)


class AppLocal:
    def __init__(self, basepath):
        self.__basepath = basepath
        self.__bc = BissClassifier(self.__basepath)
        self.__preds = None
        self.__models = self.__get_models()
        self.__modelChosen = None
        self.__predChosen = None

        self.__outputPred = Output()
        self.__outputResult = Output()

        # Define global components
        self.__resultDpdwn = Dropdown( description='Result:', disabled=False)

        # Define main Layout
        self.__mainLayout = Layout(width='20%', height='60px')
        
        self.__init_menu_predict()
        self.__init_menu_results()

    # Get list of existing models
    def __get_models(self) -> list:
        return [os.path.basename(x) for x in glob(self.__basepath+'/models/*.h5')]

    def __init_menu_predict(self):
        self.__outputPred.clear_output()

        modelDpdwn = Dropdown( options=self.__models, value=None, description='Model:',disabled=False)
        btnPredict = Button(description="Predict", button_style='primary', layout=self.__mainLayout)
        menuPredict = VBox([modelDpdwn, btnPredict])

        btnPredict.on_click(self.__on_predict_clicked)
        modelDpdwn.observe(self.__on_model_change)
        
        display(menuPredict, self.__outputPred)


    def __init_menu_results(self):
        self.__outputResult.clear_output()

        btnSave = Button(description="Save", button_style='primary', layout=self.__mainLayout)

        # Change handlers
        btnSave.on_click(self.__on_save_clicked)
        self.__resultDpdwn.observe(self.__on_predict_change)

        if self.__preds:
            self.__resultDpdwn.options = list(self.__preds.keys())
        else:
            with self.__outputResult:
                print("NO RESULT TO SHOW YET")

        menuResult = VBox([self.__resultDpdwn, btnSave])

        # Display components
        display(menuResult, self.__outputResult)

    def __on_model_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if not COLAB:
                self.__init_menu_predict()
            self.__modelChosen = change['new']
            if self.__bc.load_model(f'{self.__basepath}/models/{self.__modelChosen}'):
                with self.__outputPred:
                    print(f"Loaded model {self.__modelChosen}")
            else:
                # Unset model if could no load
                self.__modelChosen = None

    def __on_predict_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.__init_menu_results()
            self.__predChosen = change['new']
            with self.__outputResult:
                # Get result data
                brainmask = self.__bc.get_brainmask(self.__predChosen)
                image = self.__bc.get_image(self.__predChosen) * brainmask
                pred = self.__preds[self.__predChosen]

                # textBox for statistics
                items = [Label("Image:"), Label("Proportion Vessels:"), Label("Shape:")]
                left_box = VBox([items[0], items[1], items[2]])
                stat_items = [Label(self.__predChosen), \
                                Label(f'{round(get_volume_share(pred,brainmask)*100,3)} %'), \
                                Label(f'{pred.shape}')]
                right_box = VBox([stat_items[0], stat_items[1], stat_items[2]])
                display(HBox([left_box, right_box]))

                ImageSliceViewer3D(pred, image, cmap='Greys', figsize=(15,15))

    def __on_predict_clicked(self, b):
        with self.__outputPred:
            if self.__modelChosen is None:
                print("Set a model first!")
                #self.btnPredict.button_style='danger'
            else:
                spinner = yaspin()
                spinner.start()
                self.__preds = self.__bc.predict()
                spinner.stop()
                spinner.ok("✔")
                self.__init_menu_results() # Redraw results menu

    def __on_save_clicked(self, b):
        with self.__outputResult:
            path_tmp = self.__basepath + '/tmp/'
            path_tif = path_tmp + f'{self.__predChosen}_segmented.tif'

            if not os.path.exists(path_tmp):
                os.mkdir(path_tmp)

            save_prediction(path_tif, self.__preds[self.__predChosen])
            
            save_path = self.__file_dialog()
            zip(save_path, mode='w').write(path_tif, self.__predChosen + ".tif")
            print(f"Saving results to {save_path}")

            # Remove temporary path
            rmtree(path_tmp)

    # Tkinter file dialog for local notebook usage
    def __file_dialog(self):
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get location to save file to
        open_file = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[('Archive file', ['.zip'])])
        
        root.destroy()
        return open_file