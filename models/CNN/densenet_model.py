from tensorflow.keras.applications import DenseNet121
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
n_classes = 5
input_shape = (150, 150, 3)

# DenseNet121

densenet_base = DenseNet121(
    weights='imagenet', 
    include_top=False, 
    input_shape=(
        config['image_size'], 
        config['image_size'], 3),
)
densenet_base.trainable = False  

# creating my DenseNet model after integrating with the base model
model_densenet = Sequential([
    densenet_base,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')  
])

model_densenet.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),    
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training DenseNet121 model
history_densenet = model_densenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)