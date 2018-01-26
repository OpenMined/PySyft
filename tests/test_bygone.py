import bygone as by

experimentAddress = 'QmURAimSH9soLzTg9H829pqz9LKDWXKsfLCnfNEmRfQvcP'
jobs = ['Qmc7BpgxRq3sm1GdmyHU3dhktRgEbFR5ZGeK8Pxk3f81nW']
result = 'QmRivcLGSopNwQS4DjjwPV14ovvaCAiALj3zEg5B54muUK'

def testAddExperiment():
    ret = by.addExperiment(experimentAddress, jobs)
    assert(ret == 200)

def testGetJob():
    jobId = by.getAvailableJobId()
    jobAddress = by.getJob(jobId)

    assert(jobAddress == jobs[0])

def testAddResult():
    ret = by.addResult(jobAddress, result)

    assert(res == 200)

def testGetResult():
    resultAddress = by.getResults(jobAddress)

    assert(resultAddress == result)
