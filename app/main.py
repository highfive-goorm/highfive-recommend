# recommend/app/main.py
from fastapi import FastAPI, HTTPException, Path
from typing import List
import httpx
import os

from .schemas import RecommendItem, RecommendResponse
from . import cosine_recsys

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(debug=True)
PRODUCT_BASE_URL = os.environ["PRODUCT_BASE_URL"]
bulk_url = f"{PRODUCT_BASE_URL}/bulk"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCT_JSON = os.path.join(BASE_DIR, "data", "product.json")
BRAND_JSON = os.path.join(BASE_DIR, "data", "brand.json")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def get_recommendations(
    user_id: str = Path(..., description="추천 대상 사용자 ID"),
    top_n: int = 6
):
    print("start of get recommend/user_id")
    # 1) 추천 ID 리스트
    try:
        result = cosine_recsys.run_recommendation(
            user_id=user_id,
            top_n=top_n,
            product_path=PRODUCT_JSON,
            brand_path=BRAND_JSON
        )
        product_ids = result["product_id"]
    except Exception as e:
        print("[DEBUG][추천 모델 에러]", e)
        raise HTTPException(status_code=500, detail=f"추천 모델 에러: {e}")
    print("[DEBUG] product_ids:", product_ids)

    # # 2) 사용자 계정 정보 조회 (생략 가능)
    # try:
    #     async with httpx.AsyncClient() as client:
    #         USER_BASE_URL = os.environ["USER_SERVICE_BASE_URL"]
    #         user_resp = await client.get(f"{USER_BASE_URL}/{user_id}")
    #         user_resp.raise_for_status()
    #         user_data = user_resp.json()
    #         user_account = user_data.get("account", user_id)
    # except Exception:
    #     user_account = user_id
    if user_id == "b50b7a33-902f-420b-afa9-8f90b99cddf9":
        user_account = "홍길동"
    elif user_id == "guest":
        user_account = "비회원"
    else:
        user_account = user_id

    # 3) bulk 엔드포인트 호출
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                bulk_url,
                json={"product_ids": product_ids},
                timeout=10.0,
            )
            resp.raise_for_status()
            prods = resp.json()  # List[dict]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"상품 서비스 오류: {e}")

    # 4) RecommendItem → RecommendResponse 매핑
    recommends: List[RecommendItem] = []
    for p in prods:
        recommends.append(RecommendItem(
            id=p["id"],
            img_url=p.get("img_url", ""),
            name=p.get("name", ""),              # product_name → name
            brand_kor=p.get("brand_kor", ""),
            discount=p.get("discount", 0),
            price=p.get("discounted_price", 0),             # price 필드 채우기
        ))

    return RecommendResponse(
        user_account=user_account,
        recommends=recommends                  # items → recommends
    )