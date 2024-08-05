from typing import Union, TypeAlias, TypedDict
import numpy as np


from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier, StackingRegressor
from sklearn.pipeline import Pipeline

# TypeHinting MachineLearing Model Instructure
PredictionType: TypeAlias = Union[
    NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]
]
MLModelScoreType = dict[str, Union[list[float], str]]
MseType: TypeAlias = Union[float, NDArray[np.float64]]
ParameterGrids = dict[str, dict[str, list[Union[int, float, str]]]]

RegressionTrainedModels = dict[str, Union[Pipeline, StackingRegressor]]
RegressionTrainModel = dict[
    str, LinearRegression | DecisionTreeClassifier | KNeighborsClassifier | Pipeline
]
ClassifierTrainedModel = dict[
    str,
    KNeighborsClassifier
    | LogisticRegression
    | DecisionTreeClassifier
    | VotingClassifier
    | Pipeline,
]
ClassificationModel = dict[
    str, Union[KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier]
]


class ModelPerformanceDict(TypedDict):
    """각 스코어"""

    mse: float
    r2: float
    acc: float
    cm: list[str]


ModelPerformanceScore: TypeAlias = dict[str, ModelPerformanceDict]


class RegressionTrainedModelsDict(TypedDict):
    model: Union[Pipeline, StackingRegressor]


RegressionTrainedModels: TypeAlias = dict[str, RegressionTrainedModelsDict]
