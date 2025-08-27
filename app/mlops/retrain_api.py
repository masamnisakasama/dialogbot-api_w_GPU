# app/mlops/retrain_api.py　
# 実際にretrainするためのエンドポイント　

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import traceback

router = APIRouter()

# 重複解消して簡略化（retrain.py消去して一本化）
@router.post("/mlops/retrain", tags=["mlops"])
def retrain_endpoint() -> Dict[str, Any]:
    """
    再学習APIトリガー。
    app.mlops.retrain.retrain_model() を呼び、結果(JSON)を返す。
    """
    try:
        # 遅延importで循環や起動コストを回避　前に数千回呼び出されてクラッシュ経験ありの要注意エンドポイント
        from app.mlops.retrain import retrain_model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"retrain_model が見つかりません。mlops/retrain.py を確認してください。 ({e})"
        )

    try:
        result = retrain_model() 
        return {"status": "ok", "result": result if result is not None else "retrain finished"}
    except Exception as e:
        # スタックを返すのは開発時のみでいいようにする
        return {"status": "error", "detail": str(e), "traceback": traceback.format_exc()}
