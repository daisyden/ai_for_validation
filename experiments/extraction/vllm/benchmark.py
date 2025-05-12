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

def delete_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/delete"
    headers = {"Content-Type": "application/json"}
    payload = {"file_path": rag_file}

    response = requests.post(url, headers=headers, json=payload)
    return response

def upload_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/ingest"

    with open(rag_file, "rb") as f:
        files = {"files": f}
        response = requests.post(url, files=files)

    return response


delete_rag_file("all")
upload_rag_file("results.txt")

import argparse

#parser = argparse.ArgumentParser(description='PR UT failure triage')
#parser.add_argument('--artifact-name', required=True, default='Inductor-XPU-E2E-Data', help='Name of the failure list artifact to triage')
#parser.add_argument('--output-dir', default='./artifacts', help='Output directory for downloaded artifacts')
#parser.add_argument('--pr', type=int, help='The PR number')

#args = parser.parse_args()
latencies = []

#failure_list = f"{args.output_dir}/PR_{str(args.pr)}/{args.artifact_name}-{args.pr}-run-"
import csv
with open('test_case.csv', 'r') as csvfile:
    _reader = csv.reader(csvfile, quotechar=',')
    import time
    for row in _reader:
        start = time.time()
        print("#######################################\n")
        test_case = row[0] + " " + row[1]
        result = row[2]
        err_msg = "".join(row[3:])
        #message = f"unit test {test_case} got {result} with error message {err_msg}, is it a known issue no matter closed or open? If yes, please return the issue id? Who is the assignee of the issue and what are the root causes and resolutions?"
        #message = f"Unit test {test_case} returned {result} with the following error message: {err_msg}. Is this a known issue based on the RAG search results? Should all issues retrieved by RAG—regardless of whether they are closed—be considered known issues? If yes, please provide the issue ID. Also, who is the assignee of the issue, and what are the identified root causes and proposed resolutions?"
        message = f"Unit test {test_case} returned failed returned: {err_msg}."
        request = {
                "messages": message,
                "stream": False,
                "top_n": 3,
            }
        
        QnA_response=QnA(request)
        
        print("Status Code:", QnA_response.status_code)
        print("Response Body:", QnA_response.text)
        
        if QnA_response.status_code==200:
            result=QnA_response.json()["choices"][0]["message"]["content"]
            answer = result.split("</think>")[-1].strip()
            print("### Result: " + message + "\n\n" + answer)

            end = time.time()
            latencies.append(end - start)
            print("\n\n*** Latency: {} s , avg: {} s\n\n".format(end - start, sum(latencies) / len(latencies)))

