import numpy as np
import joblib
import logging
import asyncio
from typing import Union

from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline

from ai_typing import (
    PredictionType,
    MseType,
    RegressionTrainedModels,
    RegressionTrainModel,
    ModelPerformanceScore,
    ParameterGrids,
)
from model_mixin import AiModelCommonConstructionMixinClass


class RegressionEnsemble(AiModelCommonConstructionMixinClass):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)
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
                # "regressor__metric_params": self.knn_metric_param,
            },
        }
        self.trained_models: RegressionTrainedModels = {}
        self.model_performance: ModelPerformanceScore = {}

    async def train_model(
        self, name: str, model: Pipeline, param_grid: dict[str, list] = None
    ) -> None:
        """
        단일 회귀 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.

        Args:
            name (str): 모델 이름.
            model (Pipeline): 모델 파이프라인 객체.
            param_grid (Dict[str, list], optional): 하이퍼파라미터 그리드. 기본값은 None.
        """
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error"
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
        mse: MseType = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        self.model_performance[name] = {"mse": mse, "r2": r2}
        logging.info(f"모델: {name}")
        logging.info(f"평균 제곱 오차 (MSE): {mse:.2f}")
        logging.info(f"결정 계수 (R^2): {r2:.2f}")
        joblib.dump(model, f"saving_model/{name}_regression_model.pkl")
        logging.info(
            f"회귀 모델 {name}이(가) {name}_regression_model.pkl로 저장되었습니다."
        )

    async def train_models(self) -> None:
        """
        세 가지 회귀 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.
        """
        tasks = []
        for name, model in self.models.items():
            param_grid = self.param_grids.get(name, None)
            tasks.append(self.train_model(name, model, param_grid))
        await asyncio.gather(*tasks)

    async def train_ensemble_model(self) -> None:
        """
        학습된 회귀 모델을 사용하여 앙상블 회귀 모델을 비동기로 학습하고 결과를 로깅하며, 앙상블 모델을 저장함.
        """
        if not self.trained_models:
            logging.warning("학습된 모델이 없습니다. 먼저 기본 모델을 학습시키세요.")
            return

        ensemble_model = StackingRegressor(
            estimators=[
                (name, model)
                for name, model in self.trained_models.items()
                if name != "ensemble"
            ],
            final_estimator=LinearRegression(),
        )
        ensemble_model.fit(self.X_train, self.y_train)
        self.trained_models["ensemble"] = ensemble_model
        y_pred = ensemble_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        self.model_performance["ensemble"] = {"mse": mse, "r2": r2}
        logging.info("앙상블 회귀 모델")
        logging.info(f"평균 제곱 오차 (MSE): {mse:.2f}")
        logging.info(f"결정 계수 (R^2): {r2:.2f}")
        joblib.dump(ensemble_model, "saving_model/ensemble_regression_model.pkl")
        logging.info(
            "앙상블 회귀 모델이 ensemble_regression_model.pkl로 저장되었습니다."
        )

    async def get_best_model(self) -> str:
        """
        가장 성능이 좋은 모델을 선택하여 그 이름을 반환함.

        Returns:
            str: 가장 성능이 좋은 모델의 이름.
        """
        best_model_name = max(
            self.model_performance, key=lambda name: self.model_performance[name]["r2"]
        )
        best_performance = self.model_performance[best_model_name]
        logging.info(f"가장 성능이 좋은 모델: {best_model_name}")
        logging.info(f"평균 제곱 오차 (MSE): {best_performance['mse']:.2f}")
        logging.info(f"결정 계수 (R^2): {best_performance['r2']:.2f}")
        return best_model_name

    async def predict(self, X_new: np.ndarray) -> dict[str, Union[np.ndarray, str]]:
        """
        새로운 데이터에 대해 비동기로 예측 수행함.

        Args:
            X_new (np.ndarray): 예측할 새로운 데이터.

        Returns:
            Dict[str, Union[np.ndarray, str]]: 예측 결과를 담은 딕셔너리.
        """
        best_model_name = await self.get_best_model()
        if best_model_name in self.trained_models:
            model = self.trained_models[best_model_name]
            predictions = model.predict(X_new)
            return {"prediction": predictions.tolist(), "model": best_model_name}
        else:
            logging.warning("최적의 모델이 없습니다. 먼저 모델을 학습시키세요.")
            return {"prediction": [], "model": "None"}


# --------------------------------------------------------------------------------------------------------------------------------

# 사용 예시
# 당뇨병 데이터셋 로드
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


# 비동기 작업 실행
async def main() -> dict[str, np.ndarray | str]:
    ensemble = RegressionEnsemble(X, y)
    await ensemble.train_models()
    await ensemble.train_ensemble_model()
    X_new = X[:5]  # 예제로 처음 5개 데이터를 사용
    result = await ensemble.predict(X_new)

    return result


# 비동기 메인 함수 실행
result = asyncio.run(main())
