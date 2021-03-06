import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class FaceMaskData:
    def __init__(self, images_path, annotations_path, multilabelKFold=False, df_file=None):      #   saving the path of the images and the xml annotations
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.is_loaded = False
        self.multilabelKFold = multilabelKFold
        self.df_file = df_file

        self.classes = [None, 'without_mask','with_mask','mask_weared_incorrect']


    '''
        Should not be used by the user. Call to `load_data` instead.
    '''
    def _split(self, train_size=.8, drop_rate=0, seed=42):

        if not self.is_loaded:
            raise RuntimeError('Split should not be used before `load_data`')

        if train_size > 1 or train_size < 0:
            raise ValueError('Split sizes can not be greater than 1 or less than 0')

        if drop_rate > 0:
            self.images = self.images[:int(len(self.images) * (1 - drop_rate))]
            self.annotates = self.annotates[:int(len(self.annotates) * (1 - drop_rate))]
            if self.multilabelKFold:
                self.labels_encode = self.labels_encode[:int(len(self.labels_encode) * (1 - drop_rate))]


        if train_size == 1:
            if self.multilabelKFold:
                return self.images, self.labels_encode
            return self.images, self.annotates

        test_size = 1 - train_size
        if self.multilabelKFold:
            x_train, x_test, y_train, y_test, l_train, l_test = train_test_split(self.images, self.annotates, self.labels_encode, test_size=test_size, random_state=seed)
            return (x_train, y_train, l_train), (x_test, y_test, l_test)

        x_train, x_test, y_train, y_test = train_test_split(self.images, self.annotates, test_size=test_size, random_state=seed)
        return (x_train, y_train), (x_test, y_test)

    '''
        Params:
            -   train_size  -   percentage of the train data, the test size therfore will be 1-train_size.
            -   drop_rate   -   percentage of amount of random data to drop, used for code testing.
            -   seed        -   the seed to use for randomize the split of the data.
    '''
    def load_data(self, train_size=.8, drop_rate=0, seed=42):
        self.is_loaded = True

        if drop_rate > 0:
            print('\033[93m-- The data loaded using drop_rate={}, therefore not all of the data will be loaded! --\033[0m'.format(drop_rate))

        self.images = np.array( [img for img in sorted(os.listdir(self.images_path))] )
        self.annotates = np.array( [ant for ant in sorted(os.listdir(self.annotations_path))] )

        if self.multilabelKFold:
            self.labels_encode = []
            for img in self.images:
                name = img.split('.')[0]
                y_decode = self.df_file[self.df_file['file']==name]['name'].unique()
                y_encode = np.zeros(len(self.classes), dtype=np.uint8)
                for cl in y_decode:
                    class_idx = self.classes.index(cl)
                    y_encode[class_idx] = 1

                self.labels_encode.append(y_encode)

        return self._split(train_size, drop_rate, seed)


def test_data():
    imgs_path = './images'
    msks_path = './annotations'

    faceMasksData = FaceMaskData(imgs_path, msks_path)
    (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    for x_phase, y_phase in [(x_train, y_train), (x_test, y_test)]:
        for x, y in zip(x_phase, y_phase):
            print(x, y)

    print('Training contains {} samples which is {:g}% of the data'.format(len(x_train), len(x_train) * 100 / (len(x_train) + len(x_test))))
    print('Validation contains {} samples which is {:g}% of the data'.format(len(x_test), len(x_test) * 100 / (len(x_train) + len(x_test))))
    

if __name__ == '__main__':
    test_data()