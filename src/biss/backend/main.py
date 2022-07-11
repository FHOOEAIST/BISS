from .biss import BissClassifier
import sys

if __name__ == '__main__':
    # TODO: improve arg-parser!
    # TODO: Get image path as argument??

    """Args: epochs(int), createPatches(0/1), pathModel(string), patchShape('z,y,x,c')
    For cross validation: python train_biss.py crossval
    """
    args = sys.argv[1:]

    # Set the basepath according to the image directory
    basepath = '../biss_data'

    bc = BissClassifier(basepath, batch_size=64)

    if len(args) == 0:
        bc.fit()
    elif len(args) == 1:
        if (str(args[0])=='crossval'):
            bc.cross_validate()
        else:
            bc.fit(int(args[0]))
    elif len(args) == 2:
        bc.fit(int(args[0]), bool(args[1]))
    elif len(args) == 3:
        bc.fit(int(args[0]), bool(int(args[1])), str(args[2]))  
    elif len(args) == 4:
        target_shape = tuple(map(int,args[3].split(',')))
        bc.fit(int(args[0]), bool(int(args[1])), str(args[2]), target_shape)