import numpy as np
import logging
import asyncio

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from ai_typing import (
    MLModelScoreType,
    RegressionTrainedModels,
    RegressionTrainModel,
    ModelPerformanceScore,
    ParameterGrids,
)
from model_mixin import AiModelCommonConstructionMixinClass, setup_logging

setup_logging("log/regression.log")


class RegressionEnsemble(AiModelCommonConstructionMixinClass):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        초기화 함수. 데이터셋을 받아 학습 및 테스트 세트로 나누고, 회귀 모델 및 앙상블 모델을 설정함.

        Args:
            X (np.ndarray): 특징 데이터.
            y (np.ndarray): 타겟 데이터.
        """
        super().__init__(X, y, type_="regression")

        # 최적의 파라미터 값을 찾기 위해서 Pipeline 설계 입니다
        self.models: RegressionTrainModel = {
            "linear_regression": Pipeline(
                [("scaler", self.scaler), ("regressor", LinearRegression())]
            ),
            "decision_tree": Pipeline(
                [("scaler", self.scaler), ("regressor", DecisionTreeRegressor())]
            ),
            "knn": Pipeline(
                [("scaler", self.scaler), ("regressor", KNeighborsRegressor())]
            ),
        }

        # 파라미터 설정
        self.param_grids: ParameterGrids = {
            "decision_tree": {
                "regressor__criterion": ["squared_error", "friedman_mse"],
                "regressor__max_depth": self.max_depth,
                "regressor__min_samples_split": self.min_sample_split,
            },
            "knn": {
                "regressor__n_neighbors": self.n_neighbors,
                "regressor__weights": self.knn_metric,
                "regressor__metric": self.knn_distance,
            },
        }

        # 앙상블 모델 설정
        self.stacking_model = StackingRegressor(
            estimators=[
                ("lr", self.models["linear_regression"]),
                ("dt", self.models["decision_tree"]),
                ("knn", self.models["knn"]),
            ],
            final_estimator=LinearRegression(),
        )
        self.trained_models: RegressionTrainedModels = {}
        self.model_performance: ModelPerformanceScore = {}

    async def train_models(self) -> None:
        """
        세 가지 회귀 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.
        """
        tasks = [
            self.train_model(
                name=name,
                model=model,
                scoring="neg_mean_squared_error",
                param_grid=self.param_grids.get(name, None),
            )
            for name, model in self.models.items()
        ]
        await asyncio.gather(*tasks)

        # 앙상블 모델 학습
        await self.train_model(
            name="stacking",
            model=self.stacking_model,
            scoring="neg_mean_squared_error",
        )

    async def predict(self, X_new: np.ndarray) -> MLModelScoreType:
        """
        새로운 데이터에 대해 비동기로 예측 수행함.

        Args:
            X_new (np.ndarray): 예측할 새로운 데이터.

        Returns:
            MLModelScoreType: 예측 결과를 담은 딕셔너리.
        """
        if not self.best_model:
            logging.warning("모델이 학습되지 않았습니다. 먼저 모델을 학습해주세요.")
            return {"prediction": [], "model": "None"}

        logging.info(f"가장 베스트 모델은 --> {self.best_model} 입니다")
        prediction = self.best_model.predict(X_new)

        # 성능 평가 및 로깅
        y_pred = self.best_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = self.best_model.score(self.X_test, self.y_test)

        logging.info(f"모델: {self.best_model}")
        logging.info(f"평균 제곱 오차 (MSE): {mse:.2f}")
        logging.info(f"결정 계수 (R^2): {r2:.2f}")

        result: MLModelScoreType = {
            "prediction": prediction.tolist(),
            "model": type(self.best_model).__name__,
        }
        return result


# --------------------------------------------------------------------------------------------------------------------------------


# 사용 예시
async def main() -> dict[str, np.ndarray | str]:
    # 당뇨병 데이터셋 로드
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # 모델 생성 및 학습
    ensemble = RegressionEnsemble(X, y)
    await ensemble.train_models()

    X_new = X[:5]  # 예제로 처음 5개 데이터를 사용
    result = await ensemble.predict(X_new)

    return result


# 비동기 메인 함수 실행
result = asyncio.run(main())
print("회귀 모델 예측 결과:", result)
