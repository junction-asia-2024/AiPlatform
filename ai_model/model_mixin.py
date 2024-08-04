import tracemalloc

import numpy as np
import logging
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from ai_typing import (
    MLModelScoreType,
    ParameterGrids,
    MseType,
    PredictionType,
    RegressionTrainedModels,
    ClassificationModel,
)


tracemalloc.start()


# 로깅 설정
def setup_logging(log_file: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Parameters:
    log_file (str): The log file name.
    log_level (int): The logging level.
    """
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])


async def load_and_predict(model_path: str, input_data: np.ndarray) -> MLModelScoreType:
    """
    저장된 회귀 모델을 로드하고 예측을 수행하는 함수입니다.

    Parameters:
    - model_path (str): 저장된 모델의 파일 경로
    - input_data (np.ndarray): 모델에 입력할 데이터. 2D 배열 형태로 제공되어야 함.

    Returns:
    - result MLModelScoreType: 예측 결과와 모델 정보를 포함한 딕셔너리
    """
    # 모델 로드
    model = joblib.load(model_path)

    # 예측 수행
    predictions = model.predict(input_data)

    # 결과를 딕셔너리 형태로 반환
    result = {
        "prediction": predictions.tolist(),  # NumPy 배열을 리스트로 변환
        "model": f"{model_path.split('/')[1].split('_')[0]}",  # 모델 유형 (여기서는 예시로 'ensemble'로 설정)
    }

    return result


class AiModelCommonConstructionMixinClass:
    def __init__(self, X: np.ndarray, y: np.ndarray, type_: str) -> None:
        """
        초기화 함수. 데이터셋을 받아 학습 및 테스트 세트로 나눔.

        Args:
            X (np.ndarray): 특징 데이터.
            y (np.ndarray): 타겟 데이터.
        """
        self.type_ = type_
        self.trained_models: RegressionTrainedModels | ClassificationModel = {}

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.scaler = StandardScaler()

        # decision parameter
        self.max_depth = [None, 10, 20, 30]
        self.min_sample_split = [2, 5, 10]

        # KNN paramater and KNN Mahalanobis Distance Interactrion cal
        self.VI = np.linalg.inv(np.cov(self.X_train.T))
        self.n_neighbors = [3, 5, 7, 9, 10]
        self.knn_metric = ["uniform", "distance"]
        self.knn_distance = ["euclidean", "manhattan"]
        # self.knn_metric_param = [None, None, {"V": self.VI}]

    async def train_model(
        self,
        name: str,
        model: Pipeline,
        scoring: str,
        param_grid: dict[str, list] = None,
    ) -> Pipeline:
        """
        모델 비동기로 학습하고, 결과 로링 학습 모델 저장

        Args:
            name (str): 모델 이름.
            model (Pipeline): 모델 파이프라인 객체.
            param_grid (Dict[str, list], optional): 하이퍼파라미터 그리드. 기본값은 None.
        """
        if param_grid:
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, scoring=scoring, cv=5
            )

            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            logging.info(
                f"{name} 모델의 최적 하이퍼파라미터: {grid_search.best_params_}"
            )
            model = best_model
        else:
            model.fit(self.X_train, self.y_train)

        self.trained_models[name] = model
        y_pred: PredictionType = model.predict(self.X_test)
        joblib.dump(model, f"saving_model/{name}_{self.type_}_model.pkl")

        return y_pred
