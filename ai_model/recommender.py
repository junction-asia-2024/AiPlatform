import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class RecommenderSystem:
    def __init__(self, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        """
        평점 데이터와 아이템 메타데이터를 초기화

        Args:
            ratings (pd.DataFrame): 사용자-아이템 평점 데이터
            items (pd.DataFrame): 아이템 메타데이터
        """
        self.user_item_matrix, self.item_meta_matrix = self._prepare_data(
            ratings, items
        )
        self.model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model_knn.fit(csr_matrix(self.user_item_matrix.values))

    def _prepare_data(
        self, ratings: pd.DataFrame, items: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        사용자-아이템 매트릭스와 아이템 메타데이터 매트릭스를 준비

        Args:
            ratings (pd.DataFrame): 사용자-아이템 평점 데이터
            items (pd.DataFrame): 아이템 메타데이터

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 사용자-아이템 매트릭스, 아이템 메타데이터 매트릭스
        """
        user_item_matrix = ratings.pivot(
            index="user_id", columns="item_id", values="rating"
        ).fillna(0)
        item_meta_matrix = items.set_index("item_id")
        return user_item_matrix, item_meta_matrix

    def collaborative_filtering_knn(
        self, user_id: int, n_neighbors: int = 5
    ) -> list[int]:
        """
        협업 필터링 기반으로 유사한 사용자들을 찾음

        Args:
            user_id (int): 사용자 ID
            n_neighbors (int, optional): 유사한 사용자 수. 기본값은 5.

        Returns:
            list[int]: 유사한 사용자들의 ID 리스트
        """
        user_count = self.user_item_matrix.shape[0]
        if n_neighbors + 1 > user_count:
            n_neighbors = user_count - 1

        user_index = self.user_item_matrix.index.get_loc(user_id)
        distances, indices = self.model_knn.kneighbors(
            self.user_item_matrix.iloc[user_index, :].values.reshape(1, -1),
            n_neighbors=n_neighbors + 1,
        )
        similar_users = self.user_item_matrix.index[indices.flatten()[1:]].tolist()
        return similar_users

    def content_based_filtering(self, item_id: int, top_n: int = 5) -> list[int]:
        """
        콘텐츠 기반 필터링으로 유사한 아이템들을 찾음

        Args:
            item_id (int): 아이템 ID
            top_n (int, optional): 유사한 아이템 수. 기본값은 5.

        Returns:
            list[int]: 유사한 아이템들의 ID 리스트
        """
        item_features = pd.get_dummies(self.item_meta_matrix)
        item_sim_matrix = cosine_similarity(item_features)
        item_index = self.item_meta_matrix.index.get_loc(item_id)
        item_sim_scores = list(enumerate(item_sim_matrix[item_index]))
        item_sim_scores = sorted(item_sim_scores, key=lambda x: x[1], reverse=True)
        similar_items = [
            self.item_meta_matrix.index[i[0]] for i in item_sim_scores[1 : top_n + 1]
        ]
        return similar_items

    def hybrid_filtering(
        self, user_id: int, item_id: int, n_neighbors: int = 5, top_n: int = 5
    ) -> list[int]:
        """
        하이브리드 필터링으로 추천 생성

        Args:
            user_id (int): 사용자 ID
            item_id (int): 아이템 ID
            n_neighbors (int, optional): 유사한 사용자 수. 기본값은 5.
            top_n (int, optional): 추천할 아이템 수. 기본값은 5.

        Returns:
            list[int]: 추천된 아이템들의 ID 리스트
        """
        similar_users = self.collaborative_filtering_knn(user_id, n_neighbors)
        similar_users_ratings = self.user_item_matrix.loc[similar_users].mean(axis=0)
        top_items_by_similar_users = (
            similar_users_ratings.sort_values(ascending=False).index[:top_n].tolist()
        )
        similar_items = self.content_based_filtering(item_id, top_n)
        hybrid_recommendations = list(
            set(top_items_by_similar_users) | set(similar_items)
        )
        return hybrid_recommendations


# 예제... KNN이여서 성능은 일관적 데이터형식 꼭 맞출것..
if __name__ == "__main__":
    ratings_data = {
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4],
        "item_id": [1, 2, 3, 1, 2, 4, 2, 3, 1, 2, 3, 4],
        "rating": [5, 4, 3, 4, 5, 2, 2, 3, 4, 5, 3, 1],
    }
    ratings = pd.DataFrame(ratings_data)

    items_data = {
        "item_id": [1, 2, 3, 4],
        "genre": ["Action", "Comedy", "Action", "Drama"],
        "year": [2000, 2005, 2010, 2015],
    }
    items = pd.DataFrame(items_data)

    recommender = RecommenderSystem(ratings, items)

    # 특정 사용자와 아이템에 대한 하이브리드 추천
    recommendations = recommender.hybrid_filtering(user_id=1, item_id=1)
    print(f"Hybrid Recommendations for User 1 and Item 1: {recommendations}")
