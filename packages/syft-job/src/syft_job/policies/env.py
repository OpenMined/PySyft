"""What a policy script gets from the runner, and the conventions it follows.

A policy is a bash script, run.sh, that the data owner writes for an api. The
input policy runs before each call's job, the output policy after it. A Python
policy is a policy.py next to a generated run.sh that calls it. There is no
helper library: a policy only uses the environment variables, files and
conventions below.

Environment variables (both policies):

    SYFT_POLICY_PARAMS_PATH         The call's params JSON, read-only. {} when
                                    the call has no params.
    SYFT_POLICY_CALL_DIR            A copy of every file the call sent, laid
                                    out as in the job, e.g. code/params.json.
                                    These are the files the api takes by name.
    SYFT_POLICY_SUBMITTER           The data scientist's email.
    SYFT_POLICY_API_NAME            The api's name.
    SYFT_POLICY_STATE_DIR           The api's state folder. It persists between
                                    runs and only the data owner reads it. Use
                                    it for anything: logs, CSVs, several files.
    SYFT_POLICY_DEFAULT_STATE_PATH  state.json inside the state folder, {} on
                                    the first run. A ready-made spot for simple
                                    key/value state.

Environment variables (output policy only):

    SYFT_POLICY_OUTPUTS_DIR         The job's outputs/. Whatever is here after
                                    the policy finishes is shared.
    SYFT_POLICY_RECIPIENTS_PATH     recipients.json, a JSON list of the emails
                                    that receive the outputs, pre-filled with
                                    the submitter.

Conventions:

    State       Read from and write to SYFT_POLICY_DEFAULT_STATE_PATH, or any
                file in SYFT_POLICY_STATE_DIR. Nothing changes unless the policy
                writes. The runner holds a lock on the state for the whole run,
                so two calls of one api never run their policies at once.
    Deny        Exit with a non-zero code. In Python, any uncaught exception
                does this. The job is marked failed.
    Reason      The last line of stderr is what the data scientist sees, in
                policy_error.txt. The full stderr stays with the data owner.
    Call files  A call sends exactly the files the api lists by name. The
                input policy can read them in SYFT_POLICY_CALL_DIR and deny the
                call because of what they hold.
    Outputs     Only outputs/ is shared. Anything else the job writes is never
                shared, so a job and its output policy can agree on a staging
                folder next to outputs/ (e.g. output_candidates/) and the
                policy builds outputs/ from it.
    Recipients  Edit recipients.json to change who receives the outputs. An
                empty list holds them back without failing the job; the data
                owner can still share them by hand. Only approved peers receive
                outputs: any other email is skipped and logged for the owner.
"""

PARAMS_PATH = "SYFT_POLICY_PARAMS_PATH"
CALL_DIR = "SYFT_POLICY_CALL_DIR"
SUBMITTER = "SYFT_POLICY_SUBMITTER"
API_NAME = "SYFT_POLICY_API_NAME"
STATE_DIR = "SYFT_POLICY_STATE_DIR"
DEFAULT_STATE_PATH = "SYFT_POLICY_DEFAULT_STATE_PATH"
OUTPUTS_DIR = "SYFT_POLICY_OUTPUTS_DIR"
RECIPIENTS_PATH = "SYFT_POLICY_RECIPIENTS_PATH"
