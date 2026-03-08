from tensorflow.keras.applications import Xception
n_classes = 5

xception_base = Xception(
    weights='imagenet', 
    include_top=False, 
    input_shape=(
        config['image_size'], 
        config['image_size'], 3),
)
xception_base.trainable = False  

model_xception = Sequential([
    xception_base,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model_xception.compile(optimizer=Adam(learning_rate=1e-5),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Training the Xception model
history_xception = model_xception.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)