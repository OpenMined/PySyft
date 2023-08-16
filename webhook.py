from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.post("/webhook")
async def handle_webhook(request: Request):
    # Ensure it's a PR event
    event_type = request.headers.get("X-GitHub-Event")
    if event_type != "pull_request":
        return {"detail": "Not a PR event"}

    # Parse the payload
    payload = await request.json()
    action = payload.get("action")

    # For this example, let's consider PR opened, synchronize (code change), and reopened actions
    if action not in ["opened", "synchronize", "reopened"]:
        return {"detail": "Not an interesting PR action"}

    pr = payload.get("pull_request", {})
    git_hash = pr.get("head", {}).get("sha")
    branch = pr.get("head", {}).get("ref")

    # Output the information
    print(f"Git Hash: {git_hash}")
    print(f"Branch: {branch}")

    return {"git_hash": git_hash, "branch": branch}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
