"""仅供离线评测接线测试生成真实客户端格式诊断，传输替身绝不访问网络。"""
from dataclasses import replace
from email.message import Message
import json

from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient


def semantic_http_completion(raw, *, config, trace_id):
    """保留原判定文本，通过真实协议代码产生独立调用和HTTP证明，不伪装真实语义通过。"""
    class Response:
        def __init__(self):
            self.status=200;self.headers=Message()
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self):
            return json.dumps({'model':config.model,'choices':[{'finish_reason':'stop','message':{'content':raw}}]}).encode()
    # 大量离线夹具不消耗真实每日预算；其余固定思考参数仍由生产factory提供。
    client=OpenAICompatibleAgentClient(config=replace(config,max_estimated_tokens_per_day=0),urlopen=lambda *args,**kwargs:Response())
    return client.complete([],trace_id=trace_id,timeout_seconds=config.attempt_timeout_seconds)
