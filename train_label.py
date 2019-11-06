import multiprocessing
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
from generator import DataGenerator
from callback import PrAucCallback
from model import get_model
from keras_radam import RAdam


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


train_df = pd.read_csv('./input/train_label.csv')
img_2_ohe_vector = pickle_load('./input/vector')
train_folder = './input/train_images'
train_imgs_folder = './input/train_images/'
num_cores = multiprocessing.cpu_count()

if __name__ == '__main__':
    train_imgs, val_imgs = train_test_split(train_df['Image'].values,
                                            test_size=0.2,
                                            stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))),
                                            random_state=2019)

    albumentations_train = Compose([VerticalFlip(),
                                    HorizontalFlip(),
                                    Rotate(limit=20),
                                    GridDistortion()],
                                   p=1)

    data_generator_train = DataGenerator(folder_imgs=train_folder,
                                         vector=img_2_ohe_vector,
                                         images_list=train_imgs,
                                         batch_size=10,
                                         augmentation=albumentations_train)
    data_generator_train_val = DataGenerator(folder_imgs=train_folder,
                                             vector=img_2_ohe_vector,
                                             images_list=train_imgs,
                                             batch_size=10,
                                             shuffle=False)
    data_generator_val = DataGenerator(folder_imgs=train_folder,
                                       vector=img_2_ohe_vector,
                                       images_list=val_imgs,
                                       batch_size=10,
                                       shuffle=False)

    train_metric_callback = PrAucCallback(num_workers=num_cores,
                                          datagenerator=data_generator_train_val)
    val_callback = PrAucCallback(num_workers=num_cores,
                                 datagenerator=data_generator_val,
                                 stage='val')

    model = get_model()

#    for base_layer in model.layers[:-3]:
#        base_layer.trainable = False

#    model.compile(optimizer=RAdam(warmup_proportion=0.1,
#                                  min_lr=1e-5),
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
#    history_0 = model.fit_generator(generator=data_generator_train,
#                                    validation_data=data_generator_val,
#                                    epochs=20,
#                                    callbacks=[train_metric_callback,
#                                               val_callback],
#                                    workers=num_cores,
#                                    verbose=1)

    for base_layer in model.layers[:-3]:
        base_layer.trainable = True

    model.compile(optimizer=RAdam(warmup_proportion=0.1,
                                  min_lr=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history_1 = model.fit_generator(generator=data_generator_train,
                                    validation_data=data_generator_val,
                                    epochs=50,
                                    callbacks=[train_metric_callback,
                                               val_callback],
                                    workers=num_cores,
                                    verbose=1)

    pr_auc_history_train = train_metric_callback.get_pr_auc_history()
    pr_auc_history_val = val_callback.get_pr_auc_history()
#    history_0 = history_0.history
#    history_1 = history_1.history

    pickle_dump(pr_auc_history_train, './input/history/history_train')
    pickle_dump(pr_auc_history_val, './input/history/history_val')
#    history_0.to_csv('./input/history/history_0.csv')
    history_1.to_csv('./input/history/history_1.csv')    
