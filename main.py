import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs=6)
print('-------------------------')
print('   --MODEL    TRAINED--  ')
print('-------------------------')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('ACCURACY: '+str(test_acc*100)+'%')
print('-------------------------')
c=0

print('Testing prediction at position 0...')
prediction = model.predict(test_images)
for i in range(10000):
    c+=1

print('Printing test image 0...')
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print('------------------------')
print('    --TEST  RESULTS--   ')
print('------------------------')
if(np.argmax(prediction[0])==test_labels[0]):
    print('Prediction successful. Test_image 0 is '+str(np.argmax(prediction[0])))
else:
    print('Prediction unsuccesful')
    print("Predicted: "+str(np.argmax(prediction[0])))
    print("Actual value: "+str(test_labels[0]))
print('------------------------')

