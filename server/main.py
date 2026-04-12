import os
import uvicorn
from server.app import app

def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info")
    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    main()
