from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from typing import Any


class MahalanobisKNNRegressor(KNeighborsRegressor):
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        metric="mahalanobis",
        metric_params: dict[str, Any] | None = None,
    ):
        """
        Mahalanobis 거리로 KNN 회귀 모델을 구현한 클래스입니다.

        Args:
            n_neighbors (int): 이웃의 수.
            weights (str or callable): 'uniform' 또는 'distance' 또는 가중치 함수.
            metric (str): 거리 계산 방법. 'mahalanobis'로 설정.
            metric_params (dict, optional): 거리 계산에 사용할 매개변수. Mahalanobis 거리의 경우 'VI'가 필요합니다.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            metric_params=metric_params,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MahalanobisKNNRegressor:
        if self.metric == "mahalanobis" and self.metric_params is None:
            raise ValueError(
                "For Mahalanobis distance, 'metric_params' must be provided with 'VI'."
            )
        return super().fit(X, y)

    def _get_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """거리 계산을 수행하는 메서드. Mahalanobis 거리를 지원합니다."""
        if self.metric == "mahalanobis":
            return pairwise_distances(
                X, Y, metric="mahalanobis", VI=self.metric_params["VI"]
            )
        return super()._get_distances(X, Y)


# 사용 예시
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    # 데이터 로드 및 전처리
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    std = StandardScaler()
    X_scaled = std.fit_transform(X)

    # 공분산 행렬 계산 및 역행렬 구하기
    cov_matrix = np.cov(X_scaled.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # MahalanobisKNNRegressor 초기화 및 학습
    knn_mahalanobis = MahalanobisKNNRegressor(
        n_neighbors=5, metric_params={"VI": inv_cov_matrix}
    )
    knn_mahalanobis.fit(X_scaled, y)

    # 예측
    y_pred = knn_mahalanobis.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
