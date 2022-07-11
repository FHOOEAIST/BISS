from .model import get_unet3d
from .datagen import TestGenerator, TrainGenerator
from .utils import Preprocessor, reassemble_patches, post_process
import numpy as np
import time, glob, os
from pickle import FALSE,dump as pickle_dump
from shutil import rmtree

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class BissClassifier:
    def __init__(self, basepath, batch_size = 32):
        self.__basepath = basepath
        self.__batch_size = batch_size
        self.model = None
        self.__testprep = Preprocessor(basepath + '/test')
        self.__trainprep = Preprocessor(basepath + '/train')
        K.set_image_data_format("channels_last")

    # Use the patch history of Preprocessor to restore original image
    def __restore_image(self, pred, name_target):
        hist = self.__testprep.patch_history[name_target]

        pred = np.reshape(pred, hist["reshape"])
        pred = np.reshape(pred, hist["patch"])
        pred = reassemble_patches(pred, hist["cut"])
        final_result = np.zeros(hist["orig"])
        
        # TODO: Write function for result extension
        bds = hist["bounds"]
        final_result[bds["zmin"]:bds["zmax"],bds["rmin"]:bds["rmax"], bds["cmin"]:bds["cmax"]] = pred
        return final_result

    def load_model(self, path) -> bool:
        try:
            self.model = load_model(path)
            return True
        except:
            return False
        
    def predict(self, pp=True, cv=FALSE) -> dict:

        # Create patches
        self.__testprep.store_patches((32,32,32), 32, 0.5)

        # Create DataGenerator
        testgen = TestGenerator(self.__basepath + '/test/patches/', (32,32,32,1), shuffle=False, batch_size=1)

        targets = []
        for file in glob.glob(self.__basepath + '/test/raw/images/*.tif'):
            if os.name == 'nt':
                targets.append(file.split('\\')[-1].split('.tif')[0])
            else:
                targets.append(file.split('/')[-1].split('.tif')[0])
        
        # Call model.predict(datagen)
        print("Getting prediction for all patches ...")

        preds = {}
        # Make prediction for each tif-stack
        for i,t in enumerate(targets):
            print("Predicting target %s/%s: %s" %(i+1,len(targets),t))
            testgen.pick_target(t)

            # Load different models when using cross validation
            if cv:
                pathModel = self.__basepath + ('/models/model_cv_%s.h5' %t)
                self.load_model(pathModel)

            pred = self.model.predict(testgen)
            testgen.reset()

            # Reshape result to original image
            result = self.__restore_image(pred,t)

            # Convert result to uint16 and store in dictionary
            result = result*np.iinfo(np.uint16).max
            result = result.astype(np.uint16)
            preds[t] = result

        # Clear patches after prediction
        rmtree(self.__basepath + '/test/patches/')

        if pp:
            for k,pred in preds.items():
                print("Apply region growth to: %s" %k)
                preds[k] = post_process(pred)

        return preds


    def fit(self, epochs:int=10, create_patches=True, path_model=None, target_shape=(32,32,32,1)):
        
        if create_patches:
            self.__trainprep.store_patches(target_shape[:-1], step_size=target_shape[0], gauss_val=0.5)

        model = get_unet3d(target_shape, depth=2, n_base_filters=32, gamma=2)
        print(model.summary())
        
        traingen = TrainGenerator(self.__basepath + '/train/patches/', target_shape, shuffle=True, batch_size=self.__batch_size, augument=True)
        valgen = traingen.validation_split(0.2)

        # TODO where to have chechkpoint??
        checkpoint_path = ("./model_%s.h5" %(time.strftime("%Y%m%d-%H%M%S")))

        model_callbacks = [
            EarlyStopping(monitor='val_auc', patience=30, mode='min', min_delta=0.0001),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=True, save_best_only=True, mode='min', save_weights_only=False)
        ]

        model.fit(x=traingen, validation_data=valgen, epochs=epochs, callbacks=model_callbacks, verbose=1)

        if not path_model is None:
            model.save(path_model)

        return 0

    def cross_validate(self, epochs:int=10, target_shape=(32,32,32,1)):
        target_path = './crossval/'
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        self.__trainprep.store_patches(target_shape[:-1], step_size=target_shape[0], gauss_val=0.5)

        # Get name of all voluminas
        targets = []
        for file in glob.glob(self.__basepath + '/train/raw/images/*.tif'):
            if os.name == 'nt':
                targets.append(file.split('\\')[-1].split('.tif')[0])
            else:
                targets.append(file.split('/')[-1].split('.tif')[0])

        # Start actual cross validation
        for i,t in enumerate(targets):
            # Get untrained model
            model = get_unet3d(target_shape, depth=2, n_base_filters=32, gamma=2)

            # Create datagenerators with actual target volume left out
            print("Do cross validation step %s/%s. Exclude -> %s" %(i+1, len(targets),t))
            traingen = TrainGenerator(self.__basepath + '/train/patches/', target_shape, shuffle=True, batch_size=self.__batch_size, augument=True)
            traingen.mask_target(t) # Exclude the current target volume
            valgen = traingen.validation_split(val_split=0.2)

            checkpoint_path = (target_path + "model_cv_%s.h5" %(t))

            model_callbacks = [
                EarlyStopping(monitor='val_auc', patience=30, mode='min', min_delta=0.0001),
                ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=True, save_best_only=True, mode='min', save_weights_only=False)
            ]

            try:
                history = model.fit(x=traingen, validation_data=valgen, epochs=epochs, callbacks=model_callbacks, verbose=1)
                
                # Dump train history to file
                hist_path = (target_path + t + "_hist.dump")
                with open(hist_path, 'wb') as hist_targetfile:
                    pickle_dump(history.history, hist_targetfile)
            except:
                print("Could not train model. Most likely gpu ran out of memory!")
        return 0

    def get_image(self, key_image):
        return self.__testprep.get_image(key_image)

    def get_brainmask(self, key_brainmask):
        return self.__testprep.get_brainmask(key_brainmask)

