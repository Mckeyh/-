from fastapi import APIRouter


router=APIRouter(
    prefix="/api/test2",
)

@router.get("/")
@router.post("/")
async def test2():
    return {"message": "mj,mj,mj,你快回来吧","code": 200}