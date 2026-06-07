"""
Simple API Key authentication for sensitive endpoints.
"""

import os
from fastapi import Header, HTTPException


API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")


async def verify_api_key(x_api_key: str = Header(default="")):
    """
    Kiểm tra API key từ header 'X-API-Key'.
    Nếu API_SECRET_KEY chưa được cấu hình trong .env thì bỏ qua (dev mode).
    """
    if not API_SECRET_KEY:
        # Dev mode: không yêu cầu API key
        return

    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide 'X-API-Key' header."
        )
