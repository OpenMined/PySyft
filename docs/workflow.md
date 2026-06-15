# Privacy-Preserving Data Analysis Workflow

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant INB as DO Inbox
    participant OUT as DO Outbox
    participant SB as DO Syftbox
    participant DO as Data Owner

    Note over DS,DO: 1. Peer Request
    DS->>INB: add_peer("do@org.com")
    DO->>INB: approve_peer_request("ds@org.com")

    Note over DO,SB: 2. Dataset Publication
    DO->>SB: create_dataset(mock, private)
    DO->>OUT: Mock data + metadata
    DS->>OUT: sync() — pull mock data

    Note over DS: 3. Explore & Develop
    DS->>DS: Test analysis on mock data

    Note over DS,INB: 4. Job Submission
    DS->>INB: submit_python_job(code)

    Note over DO: 5. Review & Execute
    DO->>INB: sync() — receive job
    DO->>DO: Approve (or reject) & run job

    Note over DO,OUT: 6. Publish Results
    DO->>OUT: Write results to outbox

    Note over DS,OUT: 7. Retrieve Results
    DS->>OUT: sync() — pull results
    DS->>DS: Read results
```

## Workflow Steps

1. **Peer Request**: The Data Scientist requests access to the Data Owner's datasite. The Data Owner reviews and approves the request.
2. **Dataset Publication**: The Data Owner publishes a dataset with both mock (public) and private components. Mock data is placed in the outbox for Data Scientists to pull.
3. **Explore & Develop**: The Data Scientist downloads the mock data to explore the structure and test their analysis code locally.
4. **Job Submission**: The Data Scientist submits analysis code via the Data Owner's inbox.
5. **Review & Execute**: The Data Owner syncs to receive the job, reviews the code, and approves (or rejects) and runs it on private data.
6. **Publish Results**: The Data Owner writes job outputs to the outbox for the Data Scientist to pull.
7. **Retrieve Results**: The Data Scientist syncs to pull the results.
