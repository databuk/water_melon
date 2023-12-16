from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image  import ImageDataGenerator



def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model