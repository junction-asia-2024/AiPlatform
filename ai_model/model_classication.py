import logging
import numpy as np
import asyncio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from ai_typing import ClassificationModel, ParameterGrids, MLModelScoreType
from model_mixin import AiModelCommonConstructionMixinClass, setup_logging

setup_logging("log/classification.log")


class ClassificationEnsemble(AiModelCommonConstructionMixinClass):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        초기화 함수. 데이터셋을 받아 학습 및 테스트 세트로 나누고, 회귀 모델 및 앙상블 모델을 설정함.

        Args:
            X (np.ndarray): 특징 데이터.
            y (np.ndarray): 타겟 데이터.
        """
        super().__init__(X, y, type_="classification")

        # 최적의 파라미터 값을 찾기 위해서 Pipeline 설계 입니다
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

        # 파라미터 설정
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

        # 앙상블 모델 설정
        self.voting_model = VotingClassifier(
            estimators=[
                ("knn", self.models["knn"]),
                ("lr", self.models["logistic_regression"]),
                ("dt", self.models["decision_tree"]),
            ],
            voting="hard",
        )

    async def train_models(self) -> None:
        """
        세 가지 분류 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.
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
        await self.train_model(
            name="voting",
            model=self.voting_model,
            scoring="accuracy",
        )
        self.best_model = max(
            self.trained_models,
            key=lambda name: accuracy_score(
                self.y_test, self.trained_models[name].predict(self.X_test)
            ),
        )
        logging.info(f"가장 성능이 좋은 모델: {self.best_model}")

    async def predict(self, X_new: np.ndarray) -> MLModelScoreType:
        """
        학습된 모델을 사용하여 예측을 수행하는 함수입니다.

        Args:
            X_new (np.ndarray): 예측할 새로운 데이터.

        Returns:
            result MLModelScoreType: 예측 결과와 모델 정보를 포함한 딕셔너리.
        """
        if not self.best_model:
            logging.warning("모델이 학습되지 않았습니다. 먼저 모델을 학습해주세요.")
            return {"prediction": [], "model": "None"}

        logging.info(f"가장 베스트 모델은 --> {self.best_model} 입니다")
        predictions = self.trained_models[self.best_model].predict(X_new)

        # 성능 평가 및 로깅
        y_pred = self.trained_models[self.best_model].predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)

        logging.info(f"모델: {self.best_model}")
        logging.info(f"혼동 행렬:\n{cm}")
        logging.info(f"정확도: {acc:.2%}")

        result = {
            "prediction": predictions.tolist(),
            "model": self.best_model,
        }
        return result


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
    # await ensemble.train_ensemble_model()
    X_new = X[:5]  # 예제로 처음 5개 데이터를 사용
    result = await ensemble.predict(X_new)
    return result


# 비동기 메인 함수 실행
result = asyncio.run(main())
logging.info(result)
