from tensorflow.keras.optimizers.schedules import CosineDecay

cosineDecay = CosineDecay(initial_learning_rate=0.001, decay_steps=1000, alpha=0.1)
