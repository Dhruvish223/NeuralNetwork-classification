from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(set(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, verbose=0)

    y_pred = model.predict(X_test).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return acc, report
