import requests
import os

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

DEFAULT_HOST_IP = "10.112.100.138"


def QnA(request, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:8888/v1/chatqna"

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=request)
    return response


def upload_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/ingest"

    with open(rag_file, "rb") as f:
        files = {"files": f}
        response = requests.post(url, files=files)

    return response

import csv
with open('test_case.csv', 'r') as csvfile:
    import pdb
    pdb.set_trace()
    _reader = csv.reader(csvfile, quotechar=',')
    for row in _reader:
        test_case = row[1] + " " + row[2]
        result = row[3]
        err_msg = "".join(row[4:])
        message = f"unit test {test_case} got {result} with error message {err_msg}, is it a known issue? If yes, please return the issue id? Who is the assignee of the issue and what are the root causes and resolutions?"

        request = {
                "messages": message,
                "stream": False
            }
        
        QnA_response=QnA(request)
        
        print("Status Code:", QnA_response.status_code)
        print("Response Body:", QnA_response.text)
        
        if QnA_response.status_code==200:
            result=QnA_response.json()["choices"][0]["message"]["content"]
            answer = result.split("</think>")[-1].strip()
            print("### Result: " + message + "\n" + answer)
