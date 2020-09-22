import tensorflow as tf
import numpy as np

#physical_devices = tf.config.experimental_list_devices()
#print("Num Devices:", len(physical_devices))
#print(physical_devices)

# Recreate the exact same model, including its weights and the optimizer
saved_model = tf.keras.models.load_model('test_model.h5')

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data

x_test  = tf.keras.utils.normalize(x_test, axis=1)

# Evaluate the model performance
test_loss, test_acc = saved_model.evaluate(x=x_test, y=y_test)
# Print out the model accuracy
print('\nTest accuracy:', test_acc)


predictions = saved_model.predict(x_test)


y_pred_class = np.argmax(predictions, axis=1)
y_pred_class
