import GalaxyNet
import tensorflow as tf
import metrics
from GalaxyDataLoader import GalaxyDataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

dataset = GalaxyDataLoader()

X, y = dataset.load()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = GalaxyNet.galaxy_model()

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_val, y_val))

model.save_weights('galaxyNet_weights.h5')

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

predi = model.predict(X_test)

predictions = np.zeros(len(predi))

for i in range(len(predi)):
    predictions[i] = np.argmax(predi[i])

label_dict = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "In-between Round Smooth",
    4: "Cigar Shaped Smooth",
    5: "Barred Spiral",
    6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral",
    8: "Edge-on without Bulge",
    9: "Edge-on with Bulge"
}

labels = [label_dict[i] for i in range(len(label_dict))]

confusion_matrix = metrics.compute_confusion_matrix(y_test, predictions, 10)
metrics.plot_confusion_matrix(confusion_matrix, labels, "Confusion matrix")

plt.show()