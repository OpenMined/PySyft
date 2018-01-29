import requests

host = "http://localhost:3000"


def add_experiment(experimentAddress, jobAddresses):
    payload = {'experimentAddress': experimentAddress,
               'jobAddresses': jobAddresses}

    r = requests.post(f'{host}/experiment', json=payload)
    print("/experiment", r)

    return r.status_code


def get_available_job_id():
    r = requests.get(host + "/availableJobId")

    print("/availableJobId", r)

    if not 'jobId' in r.json():
        return None

    return r.json()['jobId']


def get_job():
    job_id = get_available_job_id()
    if job_id is None:
        return None

    r = requests.get(f'{host}/job/{job_id}')

    print("/job/" + job_id, r)

    return r.json()['jobAddress']


def add_result(jobAddress, resultAddress):
    payload = {'jobAddress': jobAddress, 'resultAddress': resultAddress}

    r = requests.post(host + "/result", json=payload)
    print("/result", r)

    return r.status_code


def get_result(jobAddress):
    r = requests.get(host + "/results/" + jobAddress)

    print("/results/" + jobAddress, r)
    addr = r.json()['resultAddress']
    if addr == "":
        return None

    return addr
