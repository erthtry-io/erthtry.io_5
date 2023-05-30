
BATCH_SIZE = 512
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
import heapq
import os
import tensorflow_similarity
import json
import datetime
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import optuna.visualization as vis
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import tensorflow.keras.backend as K
from collections import defaultdict
from json import loads
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Lambda,
    Dropout, BatchNormalization, Flatten, Embedding, Masking, Attention, Subtract, Multiply, Reshape
)


# In[ ]:





# In[11]:


def build_siamese_network(cat_dicts, max_imgs_all, num_filters_images=72, kernel_size=3, dense_units_for_extracting=512,
                          dense_units_for_extracting_main_image=128, embedding_dim=32, dropout_coef=0.4403, activation='relu',
                          out_layer=128, func='pearson', regular=0.000015, out_activation="relu", sim_activation="sigmoid",
                          dense_units_for_extracting_all_images=1700, dense_units_for_bert_extracting=250):
    """
    Функция для построения сети Siamese.

    Аргументы:
    Обязательные
    - cat_dicts: словарь, содержащий информацию о категориях
    - max_imgs_all: максимальное количество изображений
    Необязательные
    - num_filters_images: количество фильтров для обработки изображений
    - kernel_size: размер ядра свертки
    - dense_units_for_extracting: количество нейронов в плотном слое для извлечения признаков
    - dense_units_for_extracting_main_image: количество нейронов в плотном слое для извлечения признаков главного изображения
    - embedding_dim: размерность пространства вложения
    - dropout_coef: коэффициент отсева для слоев Dropout
    - activation: функция активации
    - out_layer: количество выходных нейронов
    - func: функция расстояния ('sigmoid', 'cos_sim', 'prod', 'euclid', 'pearson')

    Возвращает:
    - siamese_network: модель сети Siamese
    """

    # Входные слои
    vector_images_all_input = Input(shape=(max_imgs_all, 128))
    vector_main_image_input = Input(shape=(128))
    bert_vector_input = Input(shape=(64,))

    # Списки слоев для каждой категории
    input_layers = []
    mask_layers = []
    embedding_layers = []
    flatten_layers = []

    # Получение списка категорий
    cat_columns = list(cat_dicts.keys())

    # Создание слоев для каждой категории
    for column in cat_columns:
        # Количество уникальных категорий в столбце
        num_categories = len(cat_dicts[column]) + 1

        # Создание входного слоя для текущей категории
        input_layer = Input(shape=(1,))
        input_layers.append(input_layer)

        # Создание слоя маскирования
        mask_layer = Masking(mask_value=0)(input_layer)
        mask_layers.append(mask_layer)
        
        # Создание слоя вложения
        embedding_layer = Embedding(num_categories, embedding_dim, embeddings_regularizer=l2(l=regular))(mask_layer)
        embedding_layers.append(embedding_layer)

        # Создание слоя выравнивания
        flatten_layer = Flatten()(embedding_layer)
        flatten_layers.append(flatten_layer)
    if len(flatten_layers) >= 2:
        flatten_layers = Concatenate()(flatten_layers)
    dense_cat = Dense(units=dense_units_for_extracting, activation=activation, kernel_regularizer=l2(regular))(flatten_layers)
    dropout_cat = Dropout(rate=dropout_coef)(dense_cat)


    # Обработка векторов изображений
    mask_layer_images = Masking(mask_value=0.0)(vector_images_all_input)
    conv1d_all_images = Conv1D(filters=num_filters_images, kernel_size=kernel_size, activation=activation)(mask_layer_images)
    max_pool_all_images = GlobalMaxPooling1D()(conv1d_all_images)
    dense_all_1 = Dense(units=dense_units_for_extracting_all_images, activation=activation, kernel_regularizer=l2(regular))(max_pool_all_images)
    dropout_all_1 = Dropout(rate=dropout_coef)(dense_all_1)
    dense_all_2 = Dense(units=dense_units_for_extracting_all_images // 2, activation=activation, kernel_regularizer=l2(regular))(dropout_all_1)
    dropout_all_2 = Dropout(rate=dropout_coef)(dense_all_2)
    dense_all_3 = Dense(units=dense_units_for_extracting_all_images // 4, activation=activation, kernel_regularizer=l2(regular))(dropout_all_2)

    # Обработка главного изображения
    dense_main_1 = Dense(units=dense_units_for_extracting_main_image, activation=activation, kernel_regularizer=l2(regular))(vector_main_image_input)
    dropout_main_1 = Dropout(rate=dropout_coef)(dense_main_1)
    dense_main_2 = Dense(units=dense_units_for_extracting_main_image // 2, activation=activation, kernel_regularizer=l2(regular))(dropout_main_1)
    dropout_main_2 = Dropout(rate=dropout_coef)(dense_main_2)

    dense_bert = Dense(units=dense_units_for_bert_extracting, activation=activation, kernel_regularizer=l2(regular))(bert_vector_input)
    dropout_bert = Dropout(rate=dropout_coef)(dense_bert)


    # Соединение слоев обработки всех данных
    merged_inputs = Concatenate()([dropout_cat, dense_all_3, dropout_main_2, dropout_bert])


    # Обработка объединенных входов
    dense_all_1 = Dense(units=dense_units_for_extracting, activation=activation, kernel_regularizer=l2(regular))(merged_inputs)
    dropout_all_1 = Dropout(rate=dropout_coef)(dense_all_1)
    dense_all_2 = Dense(units=dense_units_for_extracting // 2, activation=activation, kernel_regularizer=l2(regular))(dropout_all_1)
    dropout_all_2 = Dropout(rate=dropout_coef)(dense_all_2)
    dense_all_3 = Dense(units=dense_units_for_extracting // 4, activation=activation, kernel_regularizer=l2(regular))(dropout_all_2)
    # Выходной слой
    output = Dense(units=out_layer, activation=out_activation)(dense_all_3)

    # Создание модели сети Siamese
    siamese_model = Model(inputs=[*input_layers, vector_images_all_input, vector_main_image_input, bert_vector_input], outputs=output)

    # Создание слоев ввода для разветвления
    input_a = [*input_layers, vector_images_all_input, vector_main_image_input, bert_vector_input]
    input_b = []

    # Создание входных слоев для второй ветви
    for input_layer in input_a:
        new_input_layer = tf.keras.layers.Input(shape=input_layer.shape[1:], name=input_layer.name + "_b")
        input_b.append(new_input_layer)

    # Разветвление сети Siamese
    processed_a = siamese_model(input_a)
    processed_b = siamese_model(input_b)

    
    if func == "cos_sim":
        distance_cos = Lambda(lambda x: tf.keras.backend.l2_normalize(tf.keras.backend.concatenate([x[0], x[1]], axis=-1), axis=-1))([processed_a, processed_b])
        distance = Dense(units=1, activation=sim_activation, name='distance')(distance_cos)
    elif func == "prod":
        distance_prod = Lambda(lambda x: x[0] * x[1])([processed_a, processed_b])
        distance = Dense(units=1, activation=sim_activation, name='distance')(distance_prod)
    elif func == "euclid":
        distance_euclid = Lambda(lambda x: tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1]), axis=-1, keepdims=True)))([processed_a, processed_b])
        distance = Dense(units=1, activation=sim_activation, name='distance')(distance_euclid)
    elif func == "pearson":
        distance_pearson = Lambda(lambda x: (x[0] - tf.keras.backend.mean(x[0])) / (tf.keras.backend.std(x[0]) + tf.keras.backend.epsilon()) * (x[1] - tf.keras.backend.mean(x[1])) / (tf.keras.backend.std(x[1]) + tf.keras.backend.epsilon()))([processed_a, processed_b])
        distance = Dense(units=1, activation=sim_activation, name='distance')(distance_pearson)
    elif func == "sigmoid":
        concatenated = Concatenate()([processed_a, processed_b])
        distance = Dense(units=1, activation=sim_activation, name='distance')(concatenated)


    # Полная модель сети Siamese
    siamese_network = Model(inputs=[*input_a, *input_b], outputs=distance)

    return siamese_network


# In[12]:


def get_max_lists_count(column):
    """
    Функция для получения максимального количества списков в столбце.

    Аргументы:
    - column: столбец данных

    Возвращает:
    - max_lists_count: максимальное количество списков в столбце
    """
    max_lists_count = 0
    for row in column:
        if row is not None:
            max_lists_count = max(max_lists_count, len(row))
    return max_lists_count


def extend_lists_and_remove_images(column, max_lists):
    """
    Функция для расширения списков в столбце и удаления излишних изображений.

    Аргументы:
    - column: столбец данных
    - max_lists: максимальное количество списков

    Возвращает:
    - new_column: новый столбец данных с расширенными списками
    """
    new_column = []
    extend_arr = np.zeros((max_lists, 128), dtype=np.float32)
    for item in column:
        if item is None:
            item = extend_arr
        else:
            item = np.array([np.array(i, dtype=np.float32) for i in item], dtype=np.float32)
            item = item[:max_lists]
            extend_len = max_lists - len(item)
            if extend_len > 0:
                extend_arr[:extend_len] = 0
                item = np.concatenate([item, extend_arr[:extend_len]], axis=0)
        new_column.append(item)
    return new_column


def main_pic_column_converter(column):
    """
    Функция для конвертации столбца.

    Аргументы:
    - column: столбец данных

    Возвращает:
    - column: преобразованный столбец данных
    """
    column = column.fillna(pd.Series([np.zeros(128)]))
    column = column.apply(lambda x: [item for sublist in x for item in sublist])
    return column

def attributes_filter(column, min_value):
    """
    Функция для фильтрации атрибутов в столбце.

    Аргументы:
    - column: столбец данных
    - min_value: минимальное значение для отбора атрибутов

    Возвращает:
    - attributes_sorted: отфильтрованный и отсортированный список атрибутов
    """
    key_sorted = defaultdict(int)
    for line in column:
        if line is not None:
            for key in line:
                key_sorted[key] += 1
    key_filtered = {key: value for key, value in key_sorted.items() if value >= min_value}
    attributes_sorted = sorted(key_filtered.items(), key=lambda x: x[1], reverse=True)
    return attributes_sorted


def attributes_counter(key, column, mode, bound):
    """
    Функция для фильтрации и сортировки значений по ключу в столбце dataframe.

    Аргументы:
    - key: ключ для фильтрации и сортировки значений
    - column: столбец данных, содержащий словари
    - mode: режим вывода ('count' для значений, встречающихся больше bound раз, 'top' для топ bound самых часто встречаемых значений)
    - bound: количество значений в режиме

    Возвращает:
    - filtered_values: отфильтрованный и отсортированный список значений
    """

    key_counts = defaultdict(int)

    for line in column:
        if line is not None and key in line:
            value = line[key][0]  # Получаем значение по ключу из списка значений
            key_counts[value] += 1

    if mode == 'count':
        filtered_values = [value for value, count in key_counts.items() if count > bound]
    elif mode == 'top':
        filtered_values = heapq.nlargest(bound, key_counts, key=key_counts.get)
    else:
        raise ValueError("Недопустимый режим вывода. Допустимые значения: 'count' и 'top'.")

    return filtered_values


def merge_and_convert_in_tensors(df, pairs):
    """
    Функция для объединения и преобразования данных в тензоры.

    Аргументы:
    - df: исходный DataFrame
    - pairs: DataFrame с парами variantid

    Возвращает:
    - [tensors_1, tensors_2]: список тензоров для каждой пары данных
    """
    merged_df1 = pd.merge(pairs, df, left_on='variantid1', right_on='variantid', how='left')
    merged_df2 = pd.merge(pairs, df, left_on='variantid2', right_on='variantid', how='left')
    merged_df1 = merged_df1.drop(columns=["variantid1", "variantid2", "variantid"])
    merged_df2 = merged_df2.drop(columns=["variantid1", "variantid2", "variantid"])
    tensors_1 = []
    tensors_2 = []
    for column in merged_df1: 
        tensors_1.append(tf.convert_to_tensor(merged_df1[column].values.tolist()))
        tensors_2.append(tf.convert_to_tensor(merged_df2[column].values.tolist()))
    return [tensors_1, tensors_2]


class JsonCallback:
    def __init__(self, filename):
        self.filename = filename
        self.trials = []

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        trial_json = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state,
            'datetime_start': trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
            'datetime_complete': trial.datetime_complete.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.trials.append(trial_json)
        self.save_trials()

    def save_trials(self):
        with open(self.filename, 'w') as f:
            json.dump(self.trials, f, indent=4)


def create_optimizer(trial):
    kwargs = {}
    optimizer_options = ["Adam", "nadam"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "Adam":
        optimizer = tf.keras.optimizers.Adam(**kwargs)
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 0.00002, 0.05, log=True)
    elif optimizer_selected == "nadam":
        kwargs["learning_rate"] = trial.suggest_float("learning_rate", 0.00002, 0.05, log=True)
        optimizer = tf.keras.optimizers.Nadam(**kwargs)
    return optimizer



def objective(trial):
    dense_units_for_extracting = trial.suggest_int('final_units', 32, 512)
    dense_units_for_extracting_all_images = trial.suggest_int('all_images', 512, 4096)
    dense_units_for_extracting_bert = trial.suggest_int('bert', 16, 4096)
    dense_units_for_extracting_main_image = trial.suggest_int('main_image', 512, 1024)
    siamese_model = build_siamese_network(cat_dicts, maximum_list_count,
                                          dense_units_for_extracting=dense_units_for_extracting,
                                          dense_units_for_extracting_main_image=dense_units_for_extracting_main_image,
                                          dense_units_for_extracting_all_images=dense_units_for_extracting_all_images,
                                          dense_units_for_bert_extracting=dense_units_for_extracting_bert)
        
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    siamese_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = 0.003), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='PR', name="Pr-AUC"),
                                                                   tf.keras.metrics.F1Score(average="macro", name='F1')])  
    history = siamese_model.fit(X_train, Y_train, epochs=20, validation_data=(X_val, Y_val), verbose=1, callbacks=[early_stopping], batch_size=BATCH_SIZE)
    Y_test_variants1 = Y_test_variants.copy()
    Y_test_variants1['target'] = [i[0] for i in Y_test.numpy().tolist()]
    Y_test_variants1['scores'] = siamese_model.predict(X_test)
    Y_test_variants1 = pd.merge(Y_test_variants1, df_cats_only, left_on='variantid1', right_on='variantid', how='left')
    tf.keras.backend.clear_session()
    return pr_auc_macro(Y_test_variants1)

def contrastive_loss(y, preds, margin=1):
	y = tf.cast(y, preds.dtype)
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	return loss


def get_new_sample(X, Y):
    Y_test_variants = pd.DataFrame()
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5)
    Y_test_variants["variantid1"] = Y_test.pop("variantid1")
    Y_test_variants["variantid2"] = Y_test.pop("variantid2")
    Y_train = Y_train.drop(columns=["variantid1", "variantid2"])
    Y_val = Y_val.drop(columns=["variantid1", "variantid2"])
    datas = []
    Y_train = tf.constant(Y_train.values.tolist(), dtype=tf.float32)
    Y_val = tf.constant(Y_val.values.tolist(), dtype=tf.float32)
    Y_test = tf.constant(Y_test.values.tolist(), dtype=tf.float32)
    for data in [X_train, X_val, X_test]:
        temp = []
        for column in data:   
            temp.append(tf.constant(data[column].values.tolist(), dtype=tf.float32))
        datas.append(temp)
    tf.keras.backend.clear_session()
    return datas[0], Y_train, datas[1], Y_val, datas[2], Y_test, Y_test_variants


def pr_auc_macro(
    target_df: pd.DataFrame,
    prec_level: float = 0.75,
    cat_column: str = "categories"
) -> float:

    df = target_df

    y_true = df["target"]
    y_pred = df["scores"]
    categories = df[cat_column]

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]
        if sum(y_true_cat) == 0:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue

        valid_mask = ~np.isnan(y_pred_cat)  # Filter out NaN values
        y_pred_cat = y_pred_cat[valid_mask]
        y_true_cat = y_true_cat[valid_mask]
        if len(np.unique(y_true_cat)) < 2:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        y = y[::-1]
        x = x[::-1]
        good_idx = np.where(y >= prec_level)[0]
        if len(good_idx) > 1:
            gt_prec_level_idx = np.arange(0, good_idx[-1] + 1)
        else:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # calculate category weight anyway
        weights.append(counts[i] / len(categories))
        # calculate PRAUC for all points where the rightest x 
        # still has required precision 
        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
        except ValueError:
            pr_aucs.append(0)
            
    return np.average(pr_aucs, weights=weights)


# In[13]:


# Чтение данных из файла Parquet в pandas DataFrame
df = pd.read_parquet('My_Shiza/train_data.parquet')

# Определение лямбда-функции для загрузки JSON-строк
load = lambda x: loads(x) if x is not None else None

# Применение функции load для преобразования JSON-строк в объекты Python в двух столбцах DataFrame
df.characteristic_attributes_mapping = df.characteristic_attributes_mapping.apply(load)
df.categories = df.categories.apply(load)


# In[14]:


# Нахождение максимального количества списков в столбце и сохранение в переменную maximum_in_all
#maximum_list_count = get_max_lists_count(df["pic_embeddings_resnet_v1"])
maximum_list_count=15
# Расширение списков в столбце 'pic_embeddings_resnet_v1' до значения maximum_in_all и удаление лишних изображений
df["pic_embeddings_resnet_v1"] = extend_lists_and_remove_images(df["pic_embeddings_resnet_v1"], maximum_list_count)

# Понижение размерности столбца 'main_pic_embeddings_resnet_v1' и заполнение пустых ячеек
df["main_pic_embeddings_resnet_v1"] = main_pic_column_converter(column=df["main_pic_embeddings_resnet_v1"])


# In[15]:


# Фильтрация столбца 'characteristic_attributes_mapping' на основе минимального значения и сортировка ключей
attributes_sorted = attributes_filter(column=df['characteristic_attributes_mapping'], min_value=150000)
key_sorted = [key for (key, value) in attributes_sorted]
least = [key for (key, value) in attributes_sorted if value <= 150000]
most = [key for (key, value) in attributes_sorted if value >= 150000]
most


# In[16]:


df_cats_raw = pd.DataFrame()
df_cats_raw['variantid'] = df['variantid']

# Токенизация значений атрибутов для каждого категориального столбца
for cat_column in most:
    keys_cat = attributes_counter(cat_column, df['characteristic_attributes_mapping'], 'count', 1000)
    temp = []
    for entry in df['characteristic_attributes_mapping']:
        if entry == None:
            temp.append("rest")
        else:
            line = entry.get(cat_column)[0] if entry.get(cat_column) != None else "rest"
            if line in keys_cat:
                temp.append(line)
            else:
                temp.append("rest")
    df_cats_raw[cat_column] = temp

for cat_column in least:
    keys_cat = attributes_counter(cat_column, df['characteristic_attributes_mapping'], 'top', 30)
    temp = []
    for entry in df['characteristic_attributes_mapping']:
        if entry is None:
            temp.append("rest")
        else:
            line = entry.get(cat_column)[0] if entry.get(cat_column) != None else "rest"
            if line in keys_cat:
                temp.append(line)
            else:
                temp.append("rest")
    df_cats_raw[cat_column] = temp

    
df_cats_only = pd.DataFrame()
df_cats_only['variantid'] = df['variantid']
key_cats = defaultdict(int)
for line in df['categories']:
    if line is not None and '3' in line:
        value = line['3']
        key_cats[value] += 1
key_cats = [value for value, count in key_cats.items() if count > 1000]

temp = []
for entry in df['categories']:
    line = entry.get('3')
    if line in key_cats:
        temp.append(line)
    else:
        temp.append("rest")
df_cats_only['categories'] = temp


# In[17]:


# Создание словаря 'cat_dicts' для хранения уникальных токенов для каждого категориального столбца
cat_dicts = {}
df_cats = pd.DataFrame()
df_cats['variantid'] = df['variantid']
# Определение лямбда-функции для присвоения числовых идентификаторов токенам
is_rest = lambda x, n: 0 if n == 'rest' else (x+1)

# Обработка каждого столбца в 'df_cats'
for column in df_cats_raw:
    if column == 'variantid':
        continue
    
    # Создание словаря категорий для текущего столбца
    categories = df_cats_raw[column].unique()
    cat_dict = {cat: is_rest(i, str(cat)) for i, cat in enumerate(categories, start=0)}
    cat_dicts[column] = cat_dict
    
    # Замена значений в столбце на числовые идентификаторы
    df_cats[column] = df_cats_raw[column].map(cat_dict)


# In[18]:


# Объединение DataFrame 'df_cats' и нужных столбцов из 'df' в новый DataFrame 'processed_df'
processed_df = df_cats.merge(df[["variantid", 'pic_embeddings_resnet_v1', 'main_pic_embeddings_resnet_v1', 'name_bert_64']], on='variantid')

# Чтение данных из файла Parquet в DataFrame 'df_pairs'
df_pairs = pd.read_parquet('My_Shiza/train_pairs.parquet')
Y = df_pairs.copy()
df_pairs = df_pairs.drop(columns=["target"])
merged_df1 = pd.merge(df_pairs, processed_df, left_on='variantid1', right_on='variantid', how='left')
merged_df2 = pd.merge(df_pairs, processed_df, left_on='variantid2', right_on='variantid', how='left')
merged_df1 = merged_df1.drop(columns=["variantid1", "variantid2", "variantid"])
merged_df2 = merged_df2.drop(columns=["variantid1", "variantid2", "variantid"])
X = pd.merge(merged_df1, merged_df2, left_index=True, right_index=True)
X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_test_variants = get_new_sample(X, Y)
del df, df_cats_raw, df_cats, processed_df, df_pairs, merged_df1, merged_df2


# In[ ]:


callback_opt = JsonCallback('trials.json')
# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction="maximize", sampler=TPESampler(), pruner=HyperbandPruner())
study.optimize(objective, n_trials=25, callbacks=[callback_opt])


# In[ ]:


study_df = study.trials_dataframe(attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'params', 'user_attrs', 'system_attrs', 'state'), multi_index=False)
study_df = study_df.sort_values(by=['value'], ascending=False)
study_df.to_csv("optuna_result2.csv")
print("Best trial:")
best_trial = study.best_trial
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

print("\nAll trials:")
for trial in study.trials:
    print("  Trial number: ", trial.number)
    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print("      {}: {}".format(key, value))
