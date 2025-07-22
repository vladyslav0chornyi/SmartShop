from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ----------- AUTH ----------
class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

# ----------- FRAME ANALYZE ----------
class FrameAnalyzeRequest(BaseModel):
    image: bytes = Field(..., description="Кадр для аналізу (base64 або raw)")
    timestamp: Optional[str] = Field(None, description="Час отримання кадру")
    camera_id: Optional[str] = Field(None, description="ID камери/джерела")
    options: Optional[Dict[str, Any]] = Field({}, description="Додаткові налаштування аналізу")

class FrameAnalyzeResponse(BaseModel):
    gender: str
    emotion: str
    clothes: Dict[str, Any]
    confidence: Optional[float] = Field(None, description="Загальна впевненість моделі")
    details: Optional[Dict[str, Any]] = Field({}, description="Детальна інформація (bounding boxes, scores, etc.)")

# ----------- STORAGE ----------
class StorageSaveRequest(BaseModel):
    file_type: str = Field(..., description="Тип (image, video, audio, log, config)")
    content: bytes = Field(..., description="Вміст файлу")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метадані: час, camera_id, user_id, etc.")

class StorageSaveResponse(BaseModel):
    path: str
    status: str
    details: Optional[str] = None

class StorageGetRequest(BaseModel):
    file_type: str
    query: Optional[str] = Field(None, description="Пошуковий запит/фільтр")
    limit: Optional[int] = Field(10, description="Кількість записів")

class StorageGetResponse(BaseModel):
    files: List[str]
    metadata: Optional[List[Dict[str, Any]]] = Field([], description="Метадані для кожного файлу")

# ----------- CONFIG ----------
class ConfigGetRequest(BaseModel):
    key: Optional[str] = None

class ConfigGetResponse(BaseModel):
    value: Any
    full_config: Optional[Dict[str, Any]] = None

class ConfigSetRequest(BaseModel):
    key: str
    value: Any

class ConfigSetResponse(BaseModel):
    status: str

# ----------- METRICS ----------
class MetricsRequest(BaseModel):
    metric_type: Optional[str] = Field(None, description="Тип метрики")
    since: Optional[str] = Field(None, description="З якого часу")

class MetricData(BaseModel):
    name: str
    value: float
    timestamp: str

class MetricsResponse(BaseModel):
    metrics: List[MetricData]

# ----------- USER ----------
class UserCreateRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    role: Optional[str] = "user"

class UserCreateResponse(BaseModel):
    user_id: str
    status: str

class UserGetRequest(BaseModel):
    user_id: Optional[str] = None

class UserGetResponse(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    role: Optional[str] = "user"
    created_at: Optional[str] = None

# ----------- PLUGIN ----------
class PluginLoadRequest(BaseModel):
    plugin_name: str
    config: Optional[Dict[str, Any]] = Field({}, description="Конфігурація плагіну")

class PluginLoadResponse(BaseModel):
    name: str
    status: str
    details: Optional[str] = None

class PluginListResponse(BaseModel):
    plugins: List[str]

# ----------- ERROR ----------
class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# ----------- GENERIC RESPONSE ----------
class StatusResponse(BaseModel):
    status: str
    details: Optional[str] = None