# MT Photos非官方人脸识别API（Nvidia显卡CUDA版）

- 仅供非商业测试
- fork自[MT-Photos/mt-photos-deepface](https://github.com/MT-Photos/mt-photos-deepface), 删除了deepface相关，添加基于[deepinsight/insightface](https://github.com/deepinsight/insightface)实现的人脸识别API
- 基于[xayane的CUDA版](https://github.com/xayane/mt-photos-insightface-unofficial)，进行了高可用性改造，如果跑xayane的版本容易崩溃可以试试此版本。
- 默认不加载模型，有请求时自动加载模型，模型崩溃自动重启恢复继续处理请求，10分钟无请求自动卸载模型。

## 模型选择

insightface提供了3种模型可供选择，从上到下精度逐渐下降，可获得更快的识别速度，默认使用buffalo_l模型

可通过环境变量 `RECOGNITION_MODEL`来自定义特征提取模型；

```python
models = [
    "antelopev2",
    "buffalo_l",
    "buffalo_s",
]
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
```

初始化时会自动下载指定模型，根据连接速度可能需要等待数分钟时间

所有模型向量长度均为默认512即可


## 安装方法

- 下载镜像

```
docker pull libook/mt-photos-insightface-unofficial:latest
```

- 创建及运行容器

```
docker run --gpus all -i -p 8066:8066 -e API_AUTH_KEY=mt_photos_ai_extra --name mt-photos-insightface-unofficial --restart="unless-stopped" libook/mt-photos-insightface-unofficial:latest
```



## 打包docker镜像

可以自行编译打包镜像
```bash
docker build  . -t mt-photos-insightface-unofficial:latest
```

### 下载源码本地运行

- 安装python **3.8版本**
- 在文件夹下执行`pip install -r requirements.txt`
- 复制`.env.example`生成`.env`文件，然后修改`.env`文件内的API_AUTH_KEY
- 执行 `python server.py` ，启动服务

看到以下日志，则说明服务已经启动成功
```bash
INFO:     Started server process [27336]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8066 (Press CTRL+C to quit)
```


## API

### /check

检测服务是否可用，及api-key是否正确

```bash
curl --location --request POST 'http://127.0.0.1:8000/check' \
--header 'api-key: api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /represent

```bash
curl --location --request POST 'http://127.0.0.1:8000/represent' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- detector_backend : "insightface",
- recognition_model : 识别模型
- result : 识别到的结果

### 返回数据示例
```json
{
  "detector_backend": "insightface",
  "recognition_model": 识别模型,
  "result": [
    {
      "embedding": [ 0.5760641694068909,... 512位向量 ],
      "facial_area": {
        "x": 212,
        "y": 112,
        "w": 179,
        "h": 250,
      },
      "face_confidence": 1.0
    }
  ]
}
```
