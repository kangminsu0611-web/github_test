from fastapi import FastAPI
from fastapi.responses import HTMLResponse # HTML 응답을 위해 이 부분을 추가합니다.

app = FastAPI()

@app.get("/", response_class=HTMLResponse) # 응답 형식을 HTML로 지정합니다.
async def read_root():
    # HTML 코드를 문자열로 만듭니다.
    html_content = """
    <html>
        <head>
            <title>Docker 테스트</title>
            <style>
                body { 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    height: 100vh; 
                    margin: 0; 
                    background-color: #282c34; 
                    color: #61dafb; 
                    font-family: sans-serif;
                }
                h1 {
                    font-size: 4rem;
                    border: 3px solid #61dafb;
                    padding: 20px 40px;
                    border-radius: 10px;
                }
            </style>
        </head>
        <body>
            <h1>🚀 Docker 서버 정상 작동 중! 🚀</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)