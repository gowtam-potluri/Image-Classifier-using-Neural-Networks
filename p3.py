from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2
import numpy as np

# Step 2: Initialising the CNN
model = Sequential()

# Step 3: Convolution
model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

# Step 4: Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 5: Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 6: Flattening
model.add(Flatten())

# Step 7: Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Step 8: Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 9: ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Step 10: Load the training Set
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Step 11: Classifier Training 
model.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 2,
                         validation_steps = 2000)

# Step 12: Convert the Model to json
model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

# Step 13: Save the weights in a seperate file
model.save_weights("./model.h5")

print("Classifier trained Successfully!")

img = cv2.imread(r'C:\Users\gowta\Desktop\2.png')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

print(model.predict_classes(img))

img = cv2.imread(r'C:\Users\gowta\Desktop\1.png')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

print(model.predict_classes(img))