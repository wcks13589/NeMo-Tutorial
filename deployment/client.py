import json

import asyncio
import aiohttp

class OpenAIRequester:
    """發送 OpenAI API 請求"""
    def __init__(self, 
                 api_key, 
                 model_name, 
                 endpoint="http://localhost:8000/v1/chat/completions",
                ):
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # self.api_urls = {url: 1.0 for url in api_urls}  # 初始每個 API 響應時間相同

    async def send_request(self, session, request):
        """發送一個 OpenAI API 請求"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request["prompt"]}],
            "temperature": request["temperature"],
            "top_p": request["top_p"],
            "max_tokens": request["max_tokens"],
        }

        async with session.post(self.endpoint, json=payload, headers=self.headers) as response:
            result = await response.json()
            request["response"] = result["choices"][0]["message"]["content"]
            
            return request

    # async def send_request(self, session, request):
    #     """選擇最快的 API 來發送請求"""
    #     fastest_url = min(self.api_urls, key=self.api_urls.get)  # 選擇響應時間最短的 API
    #     start_time = asyncio.get_event_loop().time()

    #     payload = {
    #         "model": "engine",
    #         "messages": [{"role": "user", "content": request["prompt"]}],
    #         "temperature": 0.2,
    #         "top_p": 0.7
    #     }

    #     async with session.post(self.endpoint, json=payload, headers=self.headers) as response:
    #         result = await response.json()
    #         elapsed_time = asyncio.get_event_loop().time() - start_time
    #         self.api_urls[fastest_url] = (self.api_urls[fastest_url] + elapsed_time) / 2  # 更新 API 響應時間
            
    #         request["response"] = result["choices"][0]["message"]["content"]
    #         request["url"] = fastest_url
            
    #         return request

class BatchProcessor:
    """批次處理 OpenAI API 請求"""
    def __init__(self, api_key, model_name, batch_size=4):
        self.requester = OpenAIRequester(api_key, model_name)
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)  # 限制同時執行的請求數

    async def worker(self, session, request, results):
        """處理單個請求的 worker，確保 batch_size 平行化執行"""
        async with self.semaphore:  # 限制同時處理的請求數
            response = await self.requester.send_request(session, request)
            results.append(response)  # 儲存結果

    async def process_requests(self, requests):
        """不斷維持 batch_size 的請求數"""
        results = []
        tasks = []
        async with aiohttp.ClientSession() as session:
            for request in requests:
                task = asyncio.create_task(self.worker(session, request, results))
                tasks.append(task)
                
                # 當 tasks 達到 batch_size，等其中一個完成後才補充新的
                while len(tasks) >= self.batch_size:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

            # 等待所有剩餘的請求完成
            await asyncio.gather(*tasks)
        
        return results

    def run(self, requests):
        """同步方法，讓外部程式碼可以直接呼叫來處理 batch"""
        return asyncio.run(self.process_requests(requests))

# === JSONL 儲存函數 ===
def save_to_jsonl(data, file_path):
    """將資料存入 JSONL 檔案"""
    if not data:
        print("無有效數據，未寫入檔案")
        return
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    MODEL_NAME = "engine"
    API_KEY = "token-abc123" # if API KEY is needed
    concurrency = 128
    output_file = "output.jsonl"

    requests = [
        {
            "prompt": f"這是 prompt {i}",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 128,
        }
        for i in range(1024)
    ]

    processor = BatchProcessor(API_KEY, MODEL_NAME, batch_size=concurrency)
    results = processor.run(requests)

    # 印出結果
    for res in results:
        print(f"Prompt: {res['prompt']}")
        print("Response:", res["response"])
        print("-" * 80)

    # 儲存結果
    # save_to_jsonl(results, output_file)
    # print(f"成功儲存 {len(results)} 個回應對到 {output_file}")

