import pandas as pd
import tensorflow as tf

import nlp_tools

test_df = pd.read_csv('fixtures/test.csv')
train_df = pd.read_csv('fixtures/train.csv')

re_map = {
    r'\w+': ''
}
test_dict = {k: v for k, v in test_df.items()}
test_dict.update({k: v for k, v in train_df.items()})

pipe = nlp_tools.NLPPipeline(
    nlp_tools.RegexpTrashReplaceNode(
        'trash_remover',
        source_key='text',
        target_key='text2'
    )
    | nlp_tools.RegexpNode(
        'regexp_cleaner',
        source_key='text2',
        target_key='text3',
        pattern_to_sub_dict=re_map
    )
    | nlp_tools.SpaceTokenizerNode(
        'space_token',
        source_key='text2',
        target_key='text3'
    )
    | nlp_tools.WordNLTKTokenizerNode(
        'word_nltk',
        source_key='text2',
        target_key='text4'
    )
    | nlp_tools.SentenceNLTKTokenizerNode(
        'sent_nltk',
        source_key='text2',
        target_key='text5'
    )
    | nlp_tools.SKLearnCountVectorizerNode(
        'count_vectorizer',
        source_key='text2',
        # vectorizer_settings={'analyzer': 'word', 'ngram_range': (2, 2)},
        target_key='count_vectors_train'
    )
    | nlp_tools.SKLearnCountVectorizerNode(
        'count_vectorizer_test',
        source_key='text2',
        # vectorizer_settings={'analyzer': 'word', 'ngram_range': (2, 2)},
        target_key='count_vectors_test'
    )
    | nlp_tools.MultinomialNBNode(
        'multi_nb',
        predict_data_key='count_vectors_test',
        train_target_key='target',
        train_data_key='count_vectors_train'
    )
    | nlp_tools.TensorFlowLSTMNBinaryNode(
        'tf_lstmn',
        predict_data_key='text2',
        train_target_key='target',
        train_data_key='text2'
    )
    | nlp_tools.TensorFlowCNNBinaryNode(
        'tf_cnn',
        predict_data_key='text2',
        train_target_key='target',
        train_data_key='text2'
    )
    | nlp_tools.TensorFlowCustomBinaryNode(
        'custom_tnf',
        predict_data_key='text2',
        train_target_key='target',
        train_data_key='text2',
        model_layers=[
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
        ]
    )
    | nlp_tools.BertBinaryNode(
        'custom_tnf',
        predict_data_key='text2',
        train_target_key='target',
        train_data_key='text2',
    ),
)

res = pipe.consume(test_dict)
