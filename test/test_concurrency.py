import pytest
import asyncio
import torch
from httpx import AsyncClient
from server import app

@pytest.mark.asyncio
async def test_concurrent_requests():
    async with AsyncClient(app=app, base_url="http://test") as client:
        async def send_request(prompt):
            payload = {
                "code_prompt": prompt,
                "use_quantum": True
            }
            return await client.post("/generate_code", json=payload)

        tasks = [
            send_request(f"def func_{i}(x): # concurrency test")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        for r in results:
            assert r.status_code == 200, f"Request failed with status {r.status_code}"
            response_json = r.json()
            assert "final_code" in response_json, "Response missing final_code"
    
    print("All concurrency tests passed successfully!") 