import asyncio
import httpx
from pathlib import Path

async def main():
    async with httpx.AsyncClient() as client:
        # Create dummy files
        Path("dummy.jpg").write_bytes(b"dummy image data")
        Path("dummy.csv").write_bytes(b"dummy csv data")
        
        files = {
            "sample_0_thermal_image": ("dummy.jpg", open("dummy.jpg", "rb"), "image/jpeg"),
            "sample_0_acoustic_file": ("dummy.csv", open("dummy.csv", "rb"), "text/csv")
        }
        
        # We need a running server to test. 
        # But wait, we can just start the server in the background and hit it.
        pass

asyncio.run(main())
