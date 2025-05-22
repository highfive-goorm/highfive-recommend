
import os
import numpy as np  
import pandas as pd  
# import logging       
# import argparse      

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler
)
from sklearn.metrics.pairwise import cosine_similarity
from category_encoders.count import CountEncoder


def build_preprocessor(
    discount_cols: list,
    view_buy_cols: list,
    price_cols: list,
    low_cat_cols: list,
    high_cat_cols: list
) -> ColumnTransformer:
    """
    전처리 파이프라인을 생성하여 반환합니다.

    Args:
        discount_cols (list): MinMaxScaler로 [0,1] 정규화할 컬럼 리스트
        view_buy_cols (list): 로그 변환 + RobustScaler로 이상치 영향을 완화할 컬럼 리스트
        price_cols (list): 로그 변환 + MinMaxScaler로 가격 분포를 정규화할 컬럼 리스트
        low_cat_cols (list): OneHotEncoder로 명목형 인코딩할 저카디널리티 컬럼 리스트
        high_cat_cols (list): CountEncoder(normalize=True)로 빈도 비율 인코딩할 고카디널리티 컬럼 리스트

    Returns:
        ColumnTransformer: 지정된 전처리 파이프라인이 적용된 transformer 객체
    """
    # 1) 할인율 및 순위: [0,1] 범위로 MinMaxScaler 적용
    discount_pipe = Pipeline([
        ('minmax_scaler', MinMaxScaler())
    ])

    # 2) 좋아요 및 조회수: 로그 변환 후 IQR 기반 RobustScaler 적용
    view_buy_pipe = Pipeline([
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('robust_scaler', RobustScaler())
    ])

    # 3) 가격 정보: 로그 변환 후 MinMaxScaler 적용
    price_pipe = Pipeline([
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('minmax_scaler', MinMaxScaler())
    ])

    # 4) 저카디널리티 범주형: OneHotEncoder 적용
    low_cat_pipe = Pipeline([
        ('one_hot', OneHotEncoder(
            sparse_output=False,  # 희소 행렬이 아닌 numpy 배열 반환
            handle_unknown='ignore'  # 훈련 시 없는 카테고리는 0 벡터 처리
        ))
    ])

    # 5) 고카디널리티 범주형: CountEncoder(normalize=True) 적용
    high_cat_pipe = Pipeline([
        ('count_enc', CountEncoder(
            cols=high_cat_cols,
            normalize=True,        # 비율로 변환
            handle_unknown='value',
            handle_missing='value'
        ))
    ])

    # ColumnTransformer: 각 파이프라인을 지정된 컬럼 그룹에 적용, 나머지 컬럼은 제거
    preprocessor = ColumnTransformer(
        transformers=[
            ('discount_scaling', discount_pipe,  discount_cols),
            ('viewbuy_scaling',  view_buy_pipe,  view_buy_cols),
            ('price_scaling',    price_pipe,     price_cols),
            ('low_card_ohe',     low_cat_pipe,   low_cat_cols),
            ('high_card_freq',   high_cat_pipe,  high_cat_cols),
        ],
        remainder='drop'
    )
    return preprocessor


def load_data(
    product_path: str,
    brand_path: str
) -> pd.DataFrame:
    """
    상품/브랜드 데이터를 CSV 또는 JSON으로부터 로드하여 병합한 DataFrame 반환합니다

    Args:
        product_path (str): 상품 데이터 파일 경로 (.csv 또는 .json)
        brand_path (str): 브랜드 데이터 파일 경로 (.csv 또는 .json)

    Returns:
        pd.DataFrame: product_df와 brand_df를 merge한 결과
    """

    # helper: 파일 확장자에 따라 읽기 함수 선택
    def _read_any(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(path, dtype={'category_code': str})
        elif ext == '.json':
            # 줄 단위 JSON 읽어오는 코드
            return pd.read_json(path, dtype={'category_code': str})
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        
    # 1) 상품 데이터 로드: category_code를 문자열로 읽어들여 정확한 인식 유지
    product_df = _read_any(product_path)
    # 2) 브랜드 데이터 로드
    brand_df   = _read_any(brand_path)
    # 3) 병합: product.brand_id == brand.id, left join
    df = product_df.merge(
        brand_df,
        left_on='brand_id',
        right_on='id',
        how='left',
        suffixes=('', '_brand')  # 충돌 컬럼에 '_brand' 접미사 추가
    )
    return df


def compute_feature_matrix(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> np.ndarray:
    """
    DataFrame에 전처리 파이프라인을 적용하여 feature matrix를 생성합니다.

    Args:
        df (pd.DataFrame): 입력 데이터프레임
        preprocessor (ColumnTransformer): 전처리 transformer

    Returns:
        np.ndarray: 변환된 피처 매트릭스 (n_samples × n_features)
    """
    return preprocessor.fit_transform(df)


def compute_cosine_similarity(
    feature_matrix: np.ndarray
) -> np.ndarray:
    """
    피처 매트릭스로부터 아이템 간 코사인 유사도 행렬을 계산합니다.

    Args:
        feature_matrix (np.ndarray): n_items × d 형태의 피처 매트릭스

    Returns:
        np.ndarray: n_items × n_items 형태의 유사도 행렬
    """
    return cosine_similarity(feature_matrix)


def recommend_items(
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    item_index: int,
    top_n: int = 6
) -> pd.DataFrame:
    """
    특정 아이템을 기준으로 유사도 상위 top_n개 아이템을 추천하여 반환합니다.

    Args:
        df (pd.DataFrame): 원본 상품 데이터
        cosine_sim (np.ndarray): 코사인 유사도 행렬
        item_index (int): 기준 아이템의 행 인덱스
        top_n (int): 추천 아이템 개수 (기본값: 5)

    Returns:
        pd.DataFrame: 추천된 아이템들의 행 데이터
    """
    # 1) 모든 아이템에 대한 유사도 점수 계산
    scores = list(enumerate(cosine_sim[item_index]))

    # 2) 자기 자신 제거
    scores = [(i, s) for i, s in scores if i != item_index]

    # 3) 유사도 점수 내림차순 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    # 4) 상위 top_n 인덱스 추출
    top_idx = [i for i, _ in scores[:top_n]]

    # 5) 원본 df에서 해당 인덱스 행 반환
    return df.iloc[top_idx]


def evaluate_semantic_similarity(
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    K: int = 6
) -> dict:
    """
    Precision@K 및 Recall@K 지표를 계산하여 추천 성능을 평가합니다.

    Args:
        df (pd.DataFrame): 전체 상품 데이터
        cosine_sim (np.ndarray): 코사인 유사도 행렬
        K (int): top-K 개수

    Returns:
        dict: {'Precision@K': float, 'Recall@K': float}
    """
    # ID → feature matrix 행 인덱스 매핑
    id_to_idx = {pid: idx for idx, pid in enumerate(df['id'])}
    precisions, recalls = [], []
    for _, row in df.iterrows():
        idx = id_to_idx[row['id']]
        
        # 추천 함수 호출: 해당 index 상품에 대해 top-K 추천 목록 반환
        recs = recommend_items(df, cosine_sim, item_index=idx, top_n=K)

        # 추천된 상품 중 'category_code'가 쿼리와 동일한 아이템 수 집계
        relevant = (recs['category_code'] == row['category_code']).sum()

        # 전체 데이터(df)에서 쿼리와 같은 카테고리 상품 수 (자신 제외)
        total = (df['category_code'] == row['category_code']).sum() - 1
        
        # Precision@K 계산: 관련(relevant) 상품 수 / K
        precisions.append(relevant / K)

        # Recall@K 계산: 관련 상품 수 / (같은 카테고리 전체 수)
        # 카테고리 전체 수가 0일 때 나누기 에러 방지  
        recalls.append(relevant / total if total > 0 else np.nan)
    
    # 모든 쿼리에 대한 Precision@K, Recall@K 평균값 반환
    return {f'Precision@{K}': np.nanmean(precisions), f'Recall@{K}': np.nanmean(recalls)}

def run_recommendation(
    user_id: str,
    top_n: int,
    product_path: str,
    brand_path: str
) -> dict:
    """
    사용자(user_id)에 대해 top_n개의 상품 추천 결과를 반환합니다.
    Args:
      - user_id (str): 사용자 고유 ID (시드로 사용)
      - top_n (int): 추천할 상품 개수
      - product_path, brand_path (str): 데이터 파일 경로
    Returns:
      - dict: {"user_id": ..., "product_id": [...]} 형태
    """
    # 1) 데이터 로드
    df = load_data(product_path, brand_path)

    # 2) 전처리 및 feature matrix 생성
    preprocessor = build_preprocessor(
        discount_cols=['discount', 'rank'],
        view_buy_cols=['like_count','view_count','like_count_brand'],
        price_cols=['discounted_price','price'],
        low_cat_cols=['gender','category_code'],
        high_cat_cols=['brand_eng']
    )
    feature_matrix = compute_feature_matrix(df, preprocessor)

    # 3) 코사인 유사도 계산
    cosine_sim = compute_cosine_similarity(feature_matrix)

    # 4) user_id 해시로 시드 고정 → 랜덤 인덱스 선택
    seed = int(hash(user_id) % (2**32))
    rng = np.random.default_rng(seed)
    item_index = rng.integers(0, feature_matrix.shape[0])

    # 5) 추천 실행
    recs = recommend_items(df, cosine_sim, item_index=item_index, top_n=top_n)
    return {
      "user_id": user_id,
      "product_id": recs["id"].tolist()
    }

# def main():
#     """
#     스크립트 진입점:
#     - --user_id: 추천 기준 랜덤 시드로 사용할 사용자 ID
#     - --top_n: 추천 출력 개수 지정
#     - --product_path: 상품 데이터 CSV 경로
#     - --brand_path: 브랜드 데이터 CSV 경로
#     """
#     parser = argparse.ArgumentParser(description='콘텐츠 기반 추천 시스템 실행')
#     parser.add_argument('--user_id', required=True,  help='프론트엔드에서 전달된 사용자 고유 ID')
#     parser.add_argument('--top_n', type=int, default=5, help='추천 결과 개수')
#     parser.add_argument('--product_path', required=True,  help='상품 데이터 파일 경로')
#     parser.add_argument('--brand_path', required=True,  help='브랜드 데이터 파일 경로')
#     args = parser.parse_args()

#     # 로깅 설정: INFO 레벨, 시간·레벨·메시지 표시
#     # logging.basicConfig(
#     #     level=logging.INFO,
#     #     format='%(asctime)s - %(levelname)s - %(message)s'
#     # )

#     # 1) 데이터 로드
#     df = load_data(
#         product_path=args.product_path,
#         brand_path=args.brand_path
#     )
#     # logging.info(f'Loaded data shape: {df.shape}')

#     # 2) 전처리 파이프라인 구성 및 feature matrix 생성
#     preprocessor = build_preprocessor(
#         discount_cols=['discount', 'rank'],
#         view_buy_cols=['like_count', 'view_count', 'like_count_brand'],
#         price_cols=['discounted_price', 'price'],
#         low_cat_cols=['gender', 'category_code'],
#         high_cat_cols=['brand_eng']
#     )
#     feature_matrix = compute_feature_matrix(df, preprocessor)
#     # logging.info(f'Feature matrix shape: {feature_matrix.shape}')

#     # 3) 코사인 유사도 계산
#     cosine_sim = compute_cosine_similarity(feature_matrix)
#     # logging.info('Computed cosine similarity matrix')

#     # 4) user_id 해시를 시드로 랜덤 item_index 선택
#     seed = int(hash(args.user_id) % (2**32))
#     rng = np.random.default_rng(seed)
#     item_index = rng.integers(0, feature_matrix.shape[0])
#     # logging.info(f'User="{args.user_id}" selected item_index={item_index}')

#     # 5) 추천 실행 및 결과 로깅
#     recs = recommend_items(
#         df, cosine_sim, item_index=item_index, top_n=args.top_n
#     )
#     # logging.info(f'Recommendations for user="{args.user_id}":\n{recs}')
#     rec_product_ids = recs["id"].tolist()
#     result_dict = {
#         "user_id": args.user_id,
#         "product_id": rec_product_ids
#     }

#     return result_dict
#     # print(result_dict)

# if __name__ == '__main__':
#     main()