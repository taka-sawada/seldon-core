from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from alibi_detect.cd.preprocess import UAE
from alibi_detect.cd import KSDrift
from alibi_detect.utils.saving import save_detector

NUM_WORDS = 51000
MAX_LEN=200

def get_dataset(dataset: str = 'train.csv', max_len: int = 100):

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    MAX_SEQ_LEN = 200
    SKIP_LEN = 1

    X_ref = None
    with open(dataset,'r') as f:
        for line in f:
            # textID,text,selected_text,sentiment
            if "\"" in line:
                txt = line.split("\"")[1]
            else:
                txt = line.split(",")[1]
            sentimental = line.split(",")[-1].strip()
            if sentimental == "positive":
                np_senti = np.array([1.,0.,0.]).reshape(1,-1)            
            elif sentimental == "neutral":
                np_senti = np.array([0.,1.,0.]).reshape(1,-1)            
            elif sentimental == "negative":                
                np_senti = np.array([0.,0.,1.]).reshape(1,-1)            
            else:
                np_senti = np.array([0.,1.,0.]).reshape(1,-1)            
                print("Unknown sentimental!")
    
            token_data = tokenizer.encode(txt)
            token_data.extend([PAD_INDEX] * (MAX_SEQ_LEN - len(token_data)))
            token_data = np.array(token_data)
            token_data = np.expand_dims(token_data, 0)
            # print(type(token_data))
            # print(token_data.shape)
            # print(token_data)
            if X_ref is None:
                X_ref = token_data
                Y_ref = np_senti
            else:
                X_ref = np.concatenate((X_ref, token_data))
                Y_ref = np.concatenate((Y_ref, np_senti))

    print(X_ref.shape)
    print(X_ref[1000])
    print(X_ref.max())
    print(Y_ref.shape)
    print(Y_ref[1001])
    print(Y_ref[1000])
    print(Y_ref[1002])

    return (X_ref ,Y_ref), (X_ref ,Y_ref)


def imdb_model(X: np.ndarray, num_words: int = 100, emb_dim: int = 128,
               lstm_dim: int = 128, output_dim: int = 2) -> tf.keras.Model:
    inputs = Input(shape=(X.shape[1:]), dtype=tf.float32)
    x = Embedding(num_words, emb_dim)(inputs)
    x = LSTM(lstm_dim, dropout=.5)(x)
    outputs = Dense(output_dim, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

(X_train, y_train), (X_test, y_test) = get_dataset(dataset='train.csv', max_len=MAX_LEN)
#print_sentence(X_train[0], token2word)

model = imdb_model(X=X_train, num_words=NUM_WORDS, emb_dim=256, lstm_dim=128, output_dim=3)
model.fit(X_train, y_train, batch_size=32, epochs=2,
          shuffle=True, validation_data=(X_test, y_test))

Embedding = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
emb = Embedding(X_train[:5])
print(emb.shape)

tf.random.set_seed(0)

shape = tuple(emb.shape[1:])
enc_dim=32
uae = UAE(input_layer=Embedding, shape=shape, enc_dim=enc_dim)

preprocess_kwargs = {'model': uae, 'batch_size': 128}
cd = KSDrift(
    p_val=.05,
    X_ref=X_train,
    preprocess_X_ref=True,
    preprocess_kwargs=preprocess_kwargs
)

preds_h0 = cd.predict(X_train, return_p_val=True)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds_h0['data']['is_drift']]))
print('p-value: {}'.format(preds_h0['data']['p_val']))


save_detector(cd, "./model2")
print("saved!")

