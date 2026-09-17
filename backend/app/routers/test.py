from fastapi import APIRouter



router=APIRouter(
    prefix="/api/test",
)

@router.get("/")
@router.post("/")
async def test():
    return {"message": "测试接口成功","code": 200}