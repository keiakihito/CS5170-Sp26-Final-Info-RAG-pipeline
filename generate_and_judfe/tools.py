from openai import OpenAI
import os
import requests
import json
import requests


def aliyun_inference(model_name,messages):
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content


def openai_inference(model_name,messages):

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.feidaapi.com/v1"
    )

    completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )

    return completion.choices[0].message.content

def llama_inference(model_name,messages):

    client = OpenAI(
        api_key=os.environ["LLAMA_API_KEY"],
        base_url="https://api.llama-api.com"
    )

    completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )

    return completion.choices[0].message.content

def custom_inference(model_name,messages):
    response = requests.post(
        url="https://api.aimlapi.com/chat/completions",
        headers={
            "Authorization": os.environ["AIML_API_KEY"],
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model_name,
                "messages": messages,
                "max_tokens": 200,
                "stream": False,
            }
        ),
    )
    return response.json()['choices'][0]['message']['content']

def deepseek_inference(model_name,messages):

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        stream=False
    )

    return completion.choices[0].message.content


def deepseek_r1_inference(model_name,messages):
    openai = OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return chat_completion.choices[0].message.content