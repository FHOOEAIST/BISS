import numpy as np

MAX_ = np.iinfo(np.uint16).max

def random_flip(img,y,prob=0.5):
    make_flip = np.random.random(1)

    if make_flip < prob:
        random_axis = np.random.randint(0,2)
        return ( np.flip(img, axis=random_axis) , np.flip(y, axis=random_axis) )
    return img,y

def random_brightness(img, sigma=0.1, prob=0.5):
    change_brightness = np.random.random(1)
    
    if change_brightness < prob:
        img = img.astype(np.float)

        offset = np.random.normal(0, sigma*MAX_)
        img += offset
        img = np.clip(img,0,MAX_)
    return img.astype(np.uint16)

def random_noise(img, sigma=0.05, prob=0.5):
    add_noise = np.random.random(1)

    if add_noise < prob:
        img = img.astype(np.float)

        img += np.random.normal(0, sigma*MAX_, img.shape)
        img = np.clip(img,0,MAX_)
    return img.astype(np.uint16)