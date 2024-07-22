from typing import Union
from types import MethodType

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.python.keras.layers import LeakyReLU, ReLU, ELU

# 난수 고정 
tf.random.set_seed(2024)
np.random.seed(2024)

activate_leaky_relu = LeakyReLU()
input_shape = Input(shape=(None,))  # 인풋 형태 수정

# 범용 활성화 함수 
def activation_optional(
    input_tensor: tf.Tensor, 
    alpha: float = 0.3,
    leaky_relu: bool = False, 
    relu: bool = False
) -> Union[tf.Tensor, MethodType]:
    """활성화 함수 선택 BatchNormalization -> LeakyReLU or ReLU or ELU"""
    norm: tf.Tensor = Dropout(0.25)(input_tensor)
    
    # LeakyReLU 혹은 alpha 값 활성화 시 실행 
    if leaky_relu or alpha:
        activation: tf.Tensor = LeakyReLU(alpha=alpha)(norm)
    else:
        # 그렇지 않으면 둘중 하나 실행 
        if relu:
            activation = ReLU()(norm)
        else:
            activation = ELU()(norm)
    
    return activation

def dense_arch1() -> tf.Tensor:
    """Dense 범용 아키텍처"""
    units: list[int] = [10, 15, 20]  # 유닛 수 리스트
    previous_layer: tf.Tensor = input_shape  # 첫 번째 레이어의 입력 모양

    for unit in units:  # 각 유닛 수에 대해 반복문 실행
        previous_layer = Dense(units=unit, activation=activate_leaky_relu)(previous_layer)

    bn: tf.Tensor = activation_optional(previous_layer)  # 마지막 레이어에 대해 선택적으로 활성화 함수 적용
    active_dense = Dense(units=unit, activation=activate_leaky_relu)(bn)
    flatten_later = Flatten()(active_dense)
    return flatten_later

def data_concatenate() -> models.Model:
    """앙상블 아키텍처 설계"""
    dense_output: tf.Tensor = dense_arch1()
    finally_dense: tf.Tensor = Dense(10, activation='softmax')(dense_output)
    k_model: models.Model = models.Model(inputs=input_shape, outputs=finally_dense)
    k_model.summary()
    return k_model

