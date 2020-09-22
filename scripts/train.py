import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#physical_devices = tf.config.experimental_list_devices()
#print("Num Devices:", len(physical_devices))
#print(physical_devices)

print('scott1', tf.test.is_gpu_available())
print('scott2', tf.test.is_built_with_cuda())

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test, axis=1)


#Build the model object
model = Sequential()
model.add(Flatten(input_shape = x_train[0].shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10,  activation='softmax'))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=5) # Start training process

model.summary()

model.save('test_model.h5')
