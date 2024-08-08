import numpy as np
import logging
import asyncio
from typing import Coroutine, Any

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from ai_typing import MLModelScoreType, RegressionTrainModel, ParameterGrids, MseType
from model_mixin import AiModelCommonConstructionMixinClass
from reg_utils import generate_data, setup_logging

setup_logging("log/ML/regression.log")


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
                "regressor__metric_params": self.knn_metric_param,
            },
        }

        # 앙상블 모델 설정
        self.stacking_model = StackingRegressor(
            estimators=[
                ("lr", self.models["linear_regression"]),
                ("dt", self.models["decision_tree"]),
                ("knn", self.models["knn"]),
            ],
            cv=5,
            final_estimator=LinearRegression(),
        )

    async def train_models(self) -> None:
        """
        세 가지 회귀 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.
        """
        tasks: list[Coroutine[Any, Pipeline]] = [
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

    # 새로운 predict 메서드
    async def predict(self, X_new: np.ndarray) -> MLModelScoreType:
        if not self.best_model:
            logging.warning("모델이 학습되지 않았습니다. 먼저 모델을 학습해주세요.")
            return {"prediction": [], "model": "None"}

        logging.info(
            f"가장 베스트 모델은 --> {self.best_model_name} 입니다"
        )  # 모델 이름 출력
        prediction = self.best_model.predict(X_new)

        y_pred: np.ndarray = self.best_model.predict(self.X_test_std)
        mse: MseType = mean_squared_error(self.y_test, y_pred)
        r2: float = self.best_model.score(self.X_test_std, self.y_test)

        logging.info(f"모델: {self.best_model_name}")  # 모델 이름 출력
        logging.info(f"평균 제곱 오차 (MSE): {mse:.2f}")
        logging.info(f"결정 계수 (R^2): {r2:.2f}")

        result: MLModelScoreType = {
            "prediction": prediction.tolist(),
            "model": self.best_model_name,  # 모델 이름 반환
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
    result: MLModelScoreType = await ensemble.predict(X_new)

    return result


# 비동기 메인 함수 실행
result = asyncio.run(main())
print("회귀 모델 예측 결과:", result)
