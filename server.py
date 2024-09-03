import os
import sys
import asyncio
import logging
import multiprocessing as mp
from multiprocessing import Pipe, Lock
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
import uvicorn

# 初始化日志配置
logging.basicConfig(level=logging.INFO)

# 加载环境变量
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8066"))

app = FastAPI()

process = None
parent_conn = None
lock = Lock()
stop_task = None

# 启动子进程的函数
def start_image_processor():
    import image_processor  # Import image processing module
    global parent_conn, process
    parent_conn, child_conn = Pipe()
    
    process = mp.Process(target=image_processor.process_loop, args=(child_conn,))
    process.start()

# 停止子进程的函数
def stop_image_processor():
    global process, parent_conn
    if process is not None:
        process.terminate()
        process.join()
        process = None
        parent_conn = None
        logging.info("子进程已停止")

# 异步函数，10分钟后停止子进程
async def stop_after_timeout():
    await asyncio.sleep(600)  # 600秒 = 10分钟
    with lock:
        stop_image_processor()

@app.middleware("http")
async def check_activity(request, call_next):
    global stop_task
    if stop_task:
        stop_task.cancel()

    response = await call_next(request)
    stop_task = asyncio.create_task(stop_after_timeout())
    return response

async def verify_header(api_key: str = Header(...)):
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.get("/")
async def top_info():
    return {"title": "unofficial face recognition api for mt-photos, get more info: https://github.com/kqstone/mt-photos-insightface-unofficial", "link": "https://mtmt.tech/docs/advanced/facial_api","detector_backend": "insightface", "recognition_model": "buffalo_l"}

@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {'result': 'pass'}

@app.post("/restart")
async def restart_req(api_key: str = Depends(verify_header)):
    restart_program()

@app.post("/represent")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    global process, parent_conn

    content_type = file.content_type
    image_bytes = await file.read()

    # 循环执行，直至取得结果
    while True:
        with lock:
            if process is None or not process.is_alive():
                logging.info("启动子进程")
                start_image_processor()

        try:
            parent_conn.send((image_bytes, content_type))  # 将图像字节和类型发送给子进程
            data = parent_conn.recv()  # 从子进程接收处理结果
            return data

        except Exception as e:
            #return {'result': [], 'msg': str(e)}
            # 输出错误信息
            logging.error(f"An error occurred: {e}")
            # 停止子进程，期望在下一个循环中重新启动
            stop_image_processor()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=http_port)

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)
