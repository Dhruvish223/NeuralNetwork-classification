import tensorflow as tf
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def build_model(hp, input_dim, n_classes, dropout):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(units=hp.Int('units1', 32, 256, step=32), activation='relu'))
    if dropout>0: model.add(tf.keras.layers.Dropout(dropout))
    if hp.Boolean('use_layer2'):
        model.add(tf.keras.layers.Dense(units=hp.Int('units2', 16, 128, step=16), activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def tune_and_train_model(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    tuner = RandomSearch(
        lambda hp: build_model(hp, X.shape[1], len(set(y)), params['dropout']),
        objective='val_accuracy',
        max_trials=params['max_trials'],
        executions_per_trial=1,
        directory='.',
        project_name='superstore_tuning'
    )
    tuner.search(X_train, y_train, epochs=params['epochs'], validation_split=0.2, verbose=0)
    best = tuner.get_best_models(num_models=1)[0]

    history = best.fit(X_train, y_train, epochs=params['epochs'], validation_split=0.2, verbose=0)
    y_pred = best.predict(X_test).argmax(axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    return best, history, y_test, y_pred, report
