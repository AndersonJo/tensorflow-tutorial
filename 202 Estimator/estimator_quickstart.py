"""
Tensorflow는 high-level machine learning API (tf.estimator)를 제공합니다.
Estimator를 사용하여 다양향 machine-learning 모델에 관하여 configuration, training, evaluation을 쉽게 만들어 줍니다.
해당 코드에서는 Iris Dataset 에 대해서 Estimator를 사용하여 학습을 진행합니다.
"""

import argparse
import os
from typing import List, Tuple
from urllib.request import urlopen

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import DNNClassifier

from tensorflow.contrib.learn.python.learn.datasets.base import load_csv_with_header, Dataset
from tensorflow.python.feature_column.feature_column import _NumericColumn

ROOT = os.path.abspath(os.path.dirname(__file__))

# Parse Arguments
parser = argparse.ArgumentParser(description="Iris Classification with TensorFlow's Estimator")
parser.add_argument('--datapath', default='_dataset', type=str, help='the directory path to store Iris data set')
parser = parser.parse_args()

# Constants
IRIS_TRAINING_FILE = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST_FILE = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def _download(filename, url):
    iris_dir = os.path.join(ROOT, parser.datapath)
    iris_path = os.path.join(iris_dir, filename)

    # 저장할 디렉토리 생성
    if not os.path.exists(iris_dir):
        os.mkdir(iris_dir)

    # 파일이 존재하지 않으면 다운로드
    if not os.path.exists(iris_path):
        raw = urlopen(url).read()
        with open(iris_path, 'wb') as f:
            f.write(raw)

    return iris_path


def load_iris() -> Tuple[Dataset, Dataset]:
    training_path = _download(filename=IRIS_TRAINING_FILE, url=IRIS_TRAINING_FILE)
    test_path = _download(filename=IRIS_TEST_FILE, url=IRIS_TEST_URL)

    train_dataset: Dataset = load_csv_with_header(filename=training_path, target_dtype=np.int,
                                                  features_dtype=np.float32)
    test_dataset: Dataset = load_csv_with_header(filename=test_path, target_dtype=np.int,
                                                 features_dtype=np.float32)

    return train_dataset, test_dataset


def create_model() -> DNNClassifier:
    feature_columns: List[_NumericColumn] = [tf.feature_column.numeric_column("x", shape=[4])]
    model = DNNClassifier(hidden_units=[10, 20, 10], feature_columns=feature_columns, n_classes=3,
                          optimizer='Adam',
                          model_dir='/tmp/iris_model')
    return model


def main():
    train_dataset, test_dataset = load_iris()
    model = create_model()

    # Print Dataset Information
    print('train_x:', train_dataset.data.shape)
    print('train_y:', train_dataset.target.shape)
    print('test_x:', test_dataset.data.shape)
    print('test_y:', test_dataset.target.shape)

    # Define Training Inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(train_dataset.data)},
        y=np.array(train_dataset.target),
        batch_size=512,
        num_epochs=None,
        shuffle=True
    )

    # Define Test Inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(test_dataset.data)},
        y=np.array(test_dataset.target),
        num_epochs=1,
        shuffle=False
    )

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # Train
        model.fit(input_fn=train_input_fn, steps=2000)

        # Evaluate accuracy
        print(model.evaluate(input_fn=test_input_fn))


if __name__ == '__main__':
    main()
