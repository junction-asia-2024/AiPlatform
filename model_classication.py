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

# 로깅 설정
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("ensemble_logs.log")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
MLModelScoreType = dict[str, Union[list[float], str]]


class ClassificationEnsemble:
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
            str, Union[KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier]
        ] = {
            "knn": KNeighborsClassifier(),
            "logistic_regression": LogisticRegression(),
            "decision_tree": DecisionTreeClassifier(),
        }
        self.trained_models: dict[
            str,
            Union[
                KNeighborsClassifier,
                LogisticRegression,
                DecisionTreeClassifier,
                VotingClassifier,
            ],
        ] = {}

    async def train_model(
        self,
        name: str,
        model: Union[KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier],
    ) -> None:
        """
        단일 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.

        Args:
            name (str): 모델 이름.
            model (Union[KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier]): 모델 객체.
        """
        model.fit(self.X_train, self.y_train)
        self.trained_models[name] = model
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)
        logging.info(f"모델: {name}")
        logging.info(f"혼동 행렬:\n{cm}")
        logging.info(f"정확도: {acc:.2%}")
        joblib.dump(model, f"{name}_model.pkl")
        logging.info(f"모델 {name}이(가) {name}_model.pkl로 저장되었습니다.")

    async def train_models(self) -> None:
        """
        세 가지 분류 모델을 비동기로 학습시키고 결과를 로깅하며, 학습된 모델을 저장함.
        """
        tasks = [self.train_model(name, model) for name, model in self.models.items()]
        await asyncio.gather(*tasks)

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
