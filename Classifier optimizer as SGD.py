
# coding: utf-8

# In[1]:


from keras.models import Sequential

model = Sequential()


# In[2]:


from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D


# In[3]:



model = Sequential()
model.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))


# In[13]:


#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# In[5]:


from keras.preprocessing.image import ImageDataGenerator


# In[6]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[7]:



test_datagen = ImageDataGenerator(rescale=1./255)


# In[8]:


training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(64,64),
            batch_size=32,
            class_mode='binary')


# In[9]:


test_set= test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
class_mode='binary')


# In[14]:


model.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=15,
        validation_data=test_set,
validation_steps=2000)

