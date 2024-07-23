import numpy as np
import tensorflow as tf

from keras_tuner import RandomSearch, HyperParameters
from keras import models, regularizers
from keras.src.layers import Input, Dense, Dropout, Flatten
from keras.src.layers import LeakyReLU, ReLU, ELU
from keras.src.optimizers import Adam


class RegressionModelBuilder:
    def __init__(self, hp: HyperParameters, input_dim: int) -> None:
        """
        RegressionModelBuilder 는 HyperParmeters instance 로 초기화 한다

        Args:
            hp (HyperParameters): keres_tuner 로 HyperParameter 객체 추가
        """
        self.hp = hp
        self.input_dim = input_dim

    def activation_optional(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        DropOut and activate function 를 HyperParameter 조정

        Args:
            input_tensor (tf.Tensor): Dropout 과 HyperParameter 적용 입력 Tensor

        Returns:
            tf.Tensor: Dropout과 선택된 활성화 함수가 적용된 텐서
        """
        dropout_rate: float = self.hp.Float(
            "dropout_rate", min_value=0.1, max_value=0.5, sampling="linear"
        )
        norm: tf.Tensor = Dropout(dropout_rate)(input_tensor)

        # 활성화 함수 선택
        if self.hp.Boolean("use_leaky_relu"):
            alpha: float = self.hp.Float(
                "leaky_relu_alpha", min_value=0.01, max_value=0.3, sampling="LOG"
            )
            activation: tf.Tensor = LeakyReLU(alpha=alpha)(norm)
        elif self.hp.Boolean("use_relu"):
            activation: tf.Tensor = ReLU()(norm)
        else:
            activation: tf.Tensor = ELU()(norm)

        return activation

    def dense_architecture(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        layer 수와 유닛 수를 Hyperparameter 설정하여 Dense Architecture 구축

        Returns:
            tf.Tensor: Dense Architecture Ouput Tensor

        """
        units: list[int] = [
            self.hp.Int(f"units_{i}", min_value=10, max_value=100, step=10)
            for i in range(1, 5)
        ]

        previous_layer: tf.Tensor = input_tensor

        for unit in units:
            # L1과 L2 규제를 함께 적용
            regularizer = regularizers.L1L2(
                l1=self.hp.Float("l1", min_value=1e-5, max_value=0.1, sampling="log"),
                l2=self.hp.Float("l2", min_value=1e-5, max_value=0.1, sampling="log"),
            )
            previous_layer = Dense(
                units=unit, activation=None, kernel_regularizer=regularizer
            )(previous_layer)
            previous_layer = self.activation_optional(previous_layer)

        bn: tf.Tensor = self.activation_optional(previous_layer)
        final_dense_units: int = self.hp.Int(
            "final_dense_units", min_value=10, max_value=100, step=10
        )
        active_dense: tf.Tensor = Dense(units=final_dense_units, activation=None)(bn)
        active_dense = self.activation_optional(active_dense)
        flatten_later: tf.Tensor = Flatten()(active_dense)
        return flatten_later

    def build_model(self) -> models.Model:
        """
        회귀 모델을 구축하고 컴파일

        Returns:
            models.Model: 주어진 HyperParameter 구축된 Keras 모델
        """
        # 입력 레이어 정의
        input_tensor: tf.Tensor = Input(shape=(self.input_dim,))

        # Dense 아키텍처 구축
        dense_output: tf.Tensor = self.dense_architecture(input_tensor)

        # 최종 출력 레이어
        finally_dense: tf.Tensor = Dense(1, activation=None)(dense_output)

        # 모델 정의
        model: models.Model = models.Model(inputs=input_tensor, outputs=finally_dense)

        model.compile(
            optimizer=Adam(
                self.hp.Float(
                    "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
                )
            ),
            loss="mean_squared_error",  # 회귀 문제의 손실 함수
            metrics=["mean_squared_error"],
        )

        return model


class RegressionModelTuner:
    def __init__(self, input_dim: int) -> None:
        """
        RandomSerch를 사용해서 RegressionModelTuner 초기화

        Args:
            None
        """
        self.input_dim = input_dim
        self.tuner: RandomSearch = RandomSearch(
            self.build_model,
            objective="val_loss",
            max_trials=20,
            executions_per_trial=3,
            directory="mydir",
            project_name="regression_tuning",
        )

    def build_model(self, hp: HyperParameters) -> models.Model:
        """
        하이퍼파라미터를 사용하여 모델을 구축합니다.

        Args:
            hp (HyperParameters): Keras Tuner의 하이퍼파라미터 객체

        Returns:
            models.Model: 주어진 하이퍼파라미터로 구축된 Keras 모델
        """
        builder: RegressionModelBuilder = RegressionModelBuilder(hp, self.input_dim)
        return builder.build_model()

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        회귀 모델의 하이퍼파라미터 튜닝을 수행합니다.

        Args:
            X_train (np.ndarray): 학습용 입력 데이터
            y_train (np.ndarray): 학습용 타겟 데이터
            X_val (np.ndarray): 검증용 입력 데이터
            y_val (np.ndarray): 검증용 타겟 데이터

        Returns:
            None
        """
        self.tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

        best_hps: HyperParameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print("최적의 L1 규제 값:", best_hps.get("l1"))
        print("최적의 L2 규제 값:", best_hps.get("l2"))
        print("최적의 학습률:", best_hps.get("learning_rate"))
        print("최적의 Dropout 비율:", best_hps.get("dropout_rate"))
        print("최적의 레이어 수:", best_hps.get("num_layers"))
        print("최적의 최종 Dense 유닛 수:", best_hps.get("final_dense_units"))

        best_model: models.Model = self.tuner.hypermodel.build(best_hps)
        best_model.fit(
            X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1
        )


# 데이터 생성 (예시로 랜덤 데이터 사용)
X_train: np.ndarray = np.random.rand(100, 10)
y_train: np.ndarray = np.random.rand(
    100,
)  # 회귀 문제에 맞게 랜덤 실수 값 생성
X_val: np.ndarray = np.random.rand(20, 10)
y_val: np.ndarray = np.random.rand(
    20,
)

# 모델 튜닝 및 학습
tuner: RegressionModelTuner = RegressionModelTuner(input_dim=X_train.shape[1])
tuner.tune(X_train, y_train, X_val, y_val)
