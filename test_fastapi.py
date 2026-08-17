import asyncio
from fastapi import FastAPI, Request, UploadFile
from fastapi.testclient import TestClient

app = FastAPI()

@app.post("/test")
async def test_endpoint(request: Request):
    form = await request.form()
    items = []
    for k, v in form.multi_items():
        is_file = isinstance(v, UploadFile)
        items.append(f"{k}: is_file={is_file}")
    return {"items": items}

client = TestClient(app)

def test():
    files = {
        "sample_0_thermal_image": ("dummy.jpg", b"dummy image data", "image/jpeg"),
        "sample_0_acoustic_file": ("dummy.csv", b"dummy csv data", "text/csv")
    }
    response = client.post("/test", files=files)
    print(response.json())

test()
