from pydantic import BaseModel
from typing import List

class RecommendItem(BaseModel):
    id: int
    img_url: str
    name: str            
    brand_kor: str
    discount: int
    price: int      

class RecommendResponse(BaseModel):
    user_account: str
    recommends: List[RecommendItem] 
