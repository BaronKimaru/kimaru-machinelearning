# Creating CNNs using Established CNN Models

### Do not use your CPU, use established GPUs
Problem: If you use CPU (like I did wihtout realizing at first, your model might take quite along time).
This was mine at first: 
```
Epoch 1/20118/118 ━━━━━━━━━━━━━━━━━━━━ 2236s 19s/step - accuracy: 0.3524 - loss: 2.1436 - val_accuracy: 0.5464 - val_loss: 1.3497

Epoch 2/20118/118 ━━━━━━━━━━━━━━━━━━━━ 2212s 19s/step - accuracy: 0.5016 - loss: 1.6151 - val_accuracy: 0.6115 - val_loss: 1.2527

Epoch 3/20 89/118 ━━━━━━━━━━━━━━━━━━━━ 7:27 15s/step - accuracy: 0.5230 - loss: 1.5043
```
 At 19 seconds per step, training 20 epochs actually felt like watching paint dry in slow motion. You're looking at about 35–40 minutes per epoch, which is definitely not ideal for a frozen base model.

If you are running this on a standard laptop CPU, those times are actually "normal" but painful.

Check: If you are using Google Colab or Kaggle (like me), ensure your Runtime Type is set to GPU (T4, P100, or L4).

Why it matters: Deep learning libraries like TensorFlow are optimized for the parallel processing of a GPU. A GPU will likely bring that 19s/step down to less than 1s/step.