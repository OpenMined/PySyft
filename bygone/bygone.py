import requests

host = "http://192.168.2.14:3000"


def addExperiment(experimentAddress, jobAddresses):
    payload = {'experimentAddress': experimentAddress,
               'jobAddresses': jobAddresses}

    r = requests.post(host + "/experiment", json=payload)
    print("/experiment", r)

    return r.status_code


def getAvailableJobId():
    r = requests.get(host + "/availableJobId")

    print("/availableJobId", r)

    return r.json()['jobId']


def getJob(jobId):
    r = requests.get(host + "/job/" + jobId)

    print("/job/" + jobId, r)

    return r.json()['jobAddress']


def addResult(jobAddress, resultAddress):
    payload = {'jobAddress': jobAddress, 'resultAddress': resultAddress}

    r = requests.post(host + "/result", json=payload)
    print("/result", r)

    return r.status_code


def getResults(jobAddress):
    r = requests.get(host + "/results/" + jobAddress)

    print("/results/" + jobAddress, r)

    return r.json()['resultAddress']
