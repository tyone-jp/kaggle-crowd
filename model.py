import efficientnet.keras as efn
import keras.backend as K
from keras.layers import Dense
from keras.models import Model

def get_model():
    K.clear_session()
    base_model=efn.EfficientNetB7(weights='imagenet',include_top=False,pooling='avg',input_shape=(260,260,3))
    x=base_model.output
    y_pred=Dense(4,activation='sigmoid')(x)
    return Model(inputs=base_model.input,outputs=y_pred)
