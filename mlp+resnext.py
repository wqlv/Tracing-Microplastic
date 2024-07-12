import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Concatenate
from keras.applications import ResNet50V2
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, ConfusionMatrixDisplay
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import to_categorical

tif_name = os.listdir('  ')
print(tif_name)
x1, y1 = [], []
for i in range(len(tif_name)):
    path = 'SEM_images/' + tif_name[i]
    tif_name2 = os.listdir(path)
    print(i)
    for e in range(len(tif_name2)):
        path2 = path + '/' + tif_name2[e]
        print(path2)
        img = cv2.imread(path2)
        img = cv2.resize(img, (256, 256))
        print(img.shape)
        x1.append(img / 255)
        y1.append(i)
csv_name = os.listdir('  ')
x2, y2 = [], []
print(csv_name)
max_length = 6949
for i in range(len(csv_name)):
    path = 'Spectral_data/' + csv_name[i]
    csv_name2 = os.listdir(path)
    print(i)
    for e in range(len(csv_name2)):
        path2 = path + '/' + csv_name2[e]
        print(path2)
        try:
            data = pd.read_csv(path2)
            if data.shape[0] > max_length:
                data = data.iloc[:max_length, :]
            elif data.shape[0] < max_length:
                padding = np.zeros((max_length - data.shape[0], data.shape[1]))
                data = np.vstack((data, padding))
            data = MinMaxScaler().fit_transform(data)
            x2.append(data)
            y2.append(i)
        except Exception as ex:
            print(f"Error processing file {path2}: {ex}")

x1 = np.array(x1)
y1 = np.array(y1)
x2 = np.array(x2)
y2 = np.array(y2)
y1 = to_categorical(y1, len(csv_name))

print(x1.shape, x2.shape)
print(y1)
print(y2)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.2, random_state=42, stratify=y1)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2, random_state=42, stratify=y2)

def build_model():
    inputs1 = Input(shape=(256, 256, 3))
    inputs2 = Input(shape=(6949, 2))

    x1 = ResNet50V2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))(inputs1)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    print(x1.shape)

    x2 = Flatten()(inputs2)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    print(x2.shape)

    x = Concatenate(axis=1)([x1, x2])
    print(x.shape)
    output = Dense(len(csv_name), activation='softmax')(x)
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model

model = build_model()
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_freq='epoch')
history = model.fit([x_train1, x_train2], y_train1, validation_data=([x_test1, x_test2], y_test1), epochs=1000, batch_size= , callbacks=[checkpoint])

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('Categorical Crossentropy')
plt.tight_layout()
plt.legend()
plt.show()

loss = pd.DataFrame(history.history['loss'])
val_loss = pd.DataFrame(history.history['val_loss'])
losses = pd.concat([loss, val_loss], axis=1)
losses.to_csv('log/CNNloss12.csv', index=False)

plt.figure()
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.show()

acc = pd.DataFrame(history.history['acc'])
val_acc = pd.DataFrame(history.history['val_acc'])
acces = pd.concat([acc, val_acc], axis=1)
acces.to_csv('log/CNNacc12.csv', index=False)

m = load_model(filepath)
pred = m.predict([x_test1, x_test2])
pred = np.argmax(pred, axis=-1)
y_test = np.argmax(y_test1, axis=-1)

print(classification_report(y_test, pred, digits=4))

mcc = matthews_corrcoef(y_test, pred)
print(f"Matthews Correlation Coefficient: {mcc}")

cm = confusion_matrix(y_test, pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

plt.savefig('confusion_matrix.png')
