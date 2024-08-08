import logging
import numpy as np
from numpy.typing import NDArray
from typing import Any
import pandas as pd

AmountType = NDArray[np.floating[Any]]


def generate_data(num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    데이터 생성 함수
    Args:
        num_samples (int): 생성할 샘플 수
    Returns:
        Tuple[np.ndarray, np.ndarray]: 생성된 데이터와 타겟 값
    """
    np.random.seed(42)
    financial_status = np.random.rand(num_samples) * 100
    credit_score = np.random.rand(num_samples) * 800
    revenue = np.random.rand(num_samples) * 50000
    sales = np.random.rand(num_samples) * 10000

    # 타겟 변수 계산
    loan_amount = (
        financial_status * 1.5
        + credit_score * 0.02
        + revenue * 0.01
        + sales * 0.05
        + np.random.randn(num_samples) * 1000
    )

    # 데이터 배열 생성
    data = np.stack((financial_status, credit_score, revenue, sales), axis=1)

    # 데이터 프레임 생성
    df = pd.DataFrame(
        data, columns=["Financial_Status", "Credit_Score", "Revenue", "Sales"]
    )
    df["Loan_Amount"] = loan_amount
    # CSV 파일로 저장
    df.to_csv("test.csv", index=False)

    return data, loan_amount


# 로깅 설정
def setup_logging(log_file: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Parameters:
    log_file (str): The log file name.
    log_level (int): The logging level.
    """
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
