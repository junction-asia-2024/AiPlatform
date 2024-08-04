import numpy as np
import joblib
import logging
import asyncio
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from ai_typing import ClassificationModel, ParameterGrids
from model_mixin import AiModelCommonConstructionMixinClass


class ClassificationEnsemble(AiModelCommonConstructionMixinClass):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y, type_="classification")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.models: ClassificationModel = {
            "logistic_regression": Pipeline(
                [("scaler", self.scaler), ("classifier", LogisticRegression())]
            ),
            "knn": Pipeline(
                [("scaler", self.scaler), ("classifier", KNeighborsClassifier())]
            ),
            "decision_tree": Pipeline(
                [("scaler", self.scaler), ("classifier", DecisionTreeClassifier())]
            ),
        }

        self.param_grids: ParameterGrids = {
            "decision_tree": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": self.max_depth,
                "classifier__min_samples_split": self.min_sample_split,
            },
            "knn": {
                "classifier__n_neighbors": self.n_neighbors,
                "classifier__weights": self.knn_metric,
                "classifier__metric": self.knn_distance,
                # "classifier__metric_params": self.knn_metric_param,
            },
        }

    async def train_model(
        self,
        name: str,
        model: Pipeline,
        scoring: str,
        param_grid: ClassificationModel = None,
    ) -> Pipeline:
        return await super().train_model(name, model, scoring, param_grid)

    async def train_models(self) -> None:
        tasks = []
        for name, model in self.models.items():
            param_grid = self.param_grids.get(name, None)
            tasks.append(
                self.train_model(
                    name=name, model=model, param_grid=param_grid, scoring="accuracy"
                )
            )
        return await asyncio.gather(*tasks)

    # async def train_model(
    #     self,
    #     name: str,
    #     model: Union[KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier],
    # ) -> None:
    #     """
    #     단일 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.

    #     Args:
    #         name (str): 모델 이름.
    #         model (Union[KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier]): 모델 객체.
    #     """
    #     model.fit(self.X_train, self.y_train)
    #     self.trained_models[name] = model
    #     y_pred = model.predict(self.X_test)
    #     cm = confusion_matrix(self.y_test, y_pred)
    #     acc = accuracy_score(self.y_test, y_pred)
    #     logging.info(f"모델: {name}")
    #     logging.info(f"혼동 행렬:\n{cm}")
    #     logging.info(f"정확도: {acc:.2%}")
    #     joblib.dump(model, f"{name}_model.pkl")
    #     logging.info(f"모델 {name}이(가) {name}_model.pkl로 저장되었습니다.")

    # async def train_models(self) -> None:
    #     """
    #     세 가지 분류 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.
    #     """
    #     tasks = [
    #         self.train_model(name, model, "accuracy", self.param_grids)
    #         for name, model in self.models.items()
    #     ]
    #     await asyncio.gather(*tasks)

    async def train_ensemble_model(self) -> None:
        """
        학습된 모델을 사용하여 앙상블 모델을 비동기로 학습하고 결과를 로깅하며, 앙상블 모델을 저장함.
        """

        if not self.trained_models:
            logging.warning("학습된 모델이 없습니다. 먼저 기본 모델을 학습시키세요.")
            return

        ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.trained_models.items()],
            voting="soft",
        )
        ensemble_model.fit(self.X_train, self.y_train)
        self.trained_models["ensemble"] = ensemble_model
        y_pred = ensemble_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)
        logging.info("앙상블 모델")
        logging.info(f"혼동 행렬:\n{cm}")
        logging.info(f"정확도: {acc:.2%}")
        joblib.dump(ensemble_model, "ensemble_model.pkl")
        logging.info("앙상블 모델이 ensemble_model.pkl로 저장되었습니다.")

    async def predict(self, X_new: np.ndarray) -> dict[str, Union[np.ndarray, None]]:
        """
        새로운 데이터에 대해 비동기로 예측 수행함.

        Args:
            X_new (np.ndarray): 예측할 새로운 데이터.

        Returns:
            dict[str, Union[np.ndarray, None]]: 예측 결과를 담은 딕셔너리.
        """
        if "ensemble" in self.trained_models:
            model = self.trained_models["ensemble"]
            predictions = model.predict(X_new)
            proba = model.predict_proba(X_new)
            percent = np.max(proba, axis=1) * 100
            return {"prediction": predictions, "percent": percent}
        else:
            logging.warning("앙상블 모델이 없습니다. 먼저 모델을 학습시키세요.")
            return {"prediction": None, "percent": None}


# 사용 예시
# 아이리스 데이터셋 로드
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target


# 비동기 작업 실행
async def main() -> None:
    ensemble = ClassificationEnsemble(X, y)
    await ensemble.train_models()
    await ensemble.train_ensemble_model()
    X_new = X[:5]  # 예제로 처음 5개 데이터를 사용
    result = await ensemble.predict(X_new)
    return result


# 비동기 메인 함수 실행
result = asyncio.run(main())
