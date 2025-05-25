import requests
import os
import csv

DEFAULT_HOST_IP = "10.7.180.119"


def QnA(request, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:8888/v1/chatqna"

    headers = {"Content-Type": "application/json"}

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    response = requests.post(url, headers=headers, json=request)

    os.environ["http_proxy"] = proxy 
    os.environ["https_proxy"] = proxy 

    return response

def delete_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/delete"
    headers = {"Content-Type": "application/json"}
    payload = {"file_path": rag_file}

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    response = requests.post(url, headers=headers, json=payload)

    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy

    return response

def upload_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/ingest"

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    with open(rag_file, "rb") as f:
        files = {"files": f}
        response = requests.post(url, files=files)

    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy

    return response

def upload_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/ingest"

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    with open(rag_file, "rb") as f:
        files = {"files": f}
        response = requests.post(url, files=files)

    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy


    return response


def submit_triage_request(fail_list_path, pr_number):
    import csv
    response = []
    with open(fail_list_path, 'r') as csvfile:
        _reader = csv.reader(csvfile, delimiter='|')
        err_msgs = []
        for row in _reader:
            print("#######################################\n")
            test_case = row[1].strip() + "/" + row[2].strip()
            result = row[3].strip()
            err_msg = "".join(row[4:])
            if err_msg in err_msgs:
                continue
            else:
                err_msgs.append(err_msg)

            message = f"Torch-xpu-ops pull request {pr_number} CI unit test {test_case} returned failure with error message: {err_msg}. No edidence it is a random issue."
            request = {
                    "messages": message,
                    "stream": False,
                    "top_n": 5,
                    "max_tokens": 5000,
                }
            
            QnA_response=QnA(request)
            
            print("Status Code:", QnA_response.status_code)
            print("Response Body:", QnA_response.text)
            

            if QnA_response.status_code==200:
                result=QnA_response.json()["choices"][0]["message"]["content"]
                answer = result.split("</think>")[-1].strip()
                print("### PR " +  str(pr_number) + " Result: " + message + "\n\n" + answer)
                row_to_append = row.copy()
                row_to_append.append(answer)
                response.append(row_to_append)
    return response
