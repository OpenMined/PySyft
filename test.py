import bygone as by

# test
experimentAddress = 'QmURAimSH9soLzTg9H829pqz9LKDWXKsfLCnfNEmRfQvcP'
jobs = ['Qmc7BpgxRq3sm1GdmyHU3dhktRgEbFR5ZGeK8Pxk3f81nW',
        'QmZdysXu9sXVsR4RtkT7NSx3zN7mQ3q7brKuF786MFQewb',
        'QmU4GV4NUwmiifpx34BqWPdZV1soE1PCrEtWQyPrGYpLY4']
result = 'QmRivcLGSopNwQS4DjjwPV14ovvaCAiALj3zEg5B54muUK'

by.addExperiment(experimentAddress, jobs)

jobId = by.getAvailableJobId()

jobAddress = by.getJob(jobId)

by.addResult(jobAddress, result)

print(by.getResults(jobAddress))
