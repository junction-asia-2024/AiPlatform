import numpy as np
import joblib
import logging
import asyncio
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor

# 로깅 설정
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("regression_logs.log")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


class RegressionEnsemble:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        초기화 함수. 데이터셋을 받아 학습 및 테스트 세트로 나눔.

        Args:
            X (np.ndarray): 특징 데이터.
            y (np.ndarray): 타겟 데이터.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.models: dict[
            str, Union[LinearRegression, DecisionTreeRegressor, KNeighborsRegressor]
        ] = {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(),
            "knn": KNeighborsRegressor(),
        }
        self.trained_models: dict[
            str,
            Union[
                LinearRegression,
                DecisionTreeRegressor,
                KNeighborsRegressor,
                VotingRegressor,
            ],
        ] = {}

    async def train_model(
        self,
        name: str,
        model: Union[LinearRegression, DecisionTreeRegressor, KNeighborsRegressor],
    ) -> None:
        """
        단일 회귀 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.

        Args:
            name (str): 모델 이름.
            model (Union[LinearRegression, DecisionTreeRegressor, KNeighborsRegressor]): 모델 객체.
        """
        model.fit(self.X_train, self.y_train)
        self.trained_models[name] = model
        y_pred: np.ndarray = model.predict(self.X_test)
        mse: Union[float, np.ndarray] = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
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
        tasks = [self.train_model(name, model) for name, model in self.models.items()]
        await asyncio.gather(*tasks)

    async def train_ensemble_model(self) -> None:
        """
        학습된 회귀 모델을 사용하여 앙상블 회귀 모델을 비동기로 학습하고 결과를 로깅하며, 앙상블 모델을 저장함.
        """
        if not self.trained_models:
            logging.warning("학습된 모델이 없습니다. 먼저 기본 모델을 학습시키세요.")
            return

        # estimators TypeHint --> list[tuple[str, BaseEstimator]]
        ensemble_model = VotingRegressor(
            estimators=[
                (name, model)
                for name, model in self.trained_models.items()
                if name != "ensemble"
            ],
            n_jobs=-1,
        )
        ensemble_model.fit(self.X_train, self.y_train)
        self.trained_models["ensemble"] = ensemble_model
        y_pred = ensemble_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        logging.info("앙상블 회귀 모델")
        logging.info(f"평균 제곱 오차 (MSE): {mse:.2f}")
        logging.info(f"결정 계수 (R^2): {r2:.2f}")
        joblib.dump(ensemble_model, "saving_model/ensemble_regression_model.pkl")
        logging.info(
            "앙상블 회귀 모델이 ensemble_regression_model.pkl로 저장되었습니다."
        )

    async def predict(self, X_new: np.ndarray) -> dict[str, np.ndarray]:
        """
        새로운 데이터에 대해 비동기로 예측 수행함.

        Args:
            X_new (np.ndarray): 예측할 새로운 데이터.

        Returns:
            dict[str, np.ndarray]: 예측 결과를 담은 딕셔너리.
        """
        if "ensemble" in self.trained_models:
            model = self.trained_models["ensemble"]
            predictions = model.predict(X_new)
            return {"prediction": predictions}
        else:
            logging.warning("앙상블 모델이 없습니다. 먼저 모델을 학습시키세요.")
            return {"prediction": np.array([])}


# 사용 예시
# 당뇨병 데이터셋 로드
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


# 비동기 작업 실행
async def main():
    ensemble = RegressionEnsemble(X, y)
    await ensemble.train_models()
    await ensemble.train_ensemble_model()
    X_new = X[:5]  # 예제로 처음 5개 데이터를 사용
    result = await ensemble.predict(X_new)
    return result


# 비동기 메인 함수 실행
asyncio.run(main())
