from typing import Any, List

from syft_client.sync.connections.drive.gdrive_retry import execute_with_retries

GDRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def listify(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def gather_all_file_and_folder_ids_recursive(service, folder_id) -> List[str]:
    """
    Gather all file and folder IDs recursively.

    NOTE: Due to Google Drive eventual consistency, recently created files may
    not appear in query results. For deletion, it's safer to just delete the
    parent folder and rely on Google Drive's cascade deletion.
    """
    res = set([folder_id])
    query = f"'{folder_id}' in parents"
    results = execute_with_retries(
        service.files().list(q=query, fields="files(id, name, mimeType)")
    )
    for item in results.get("files", []):
        if item["mimeType"] == GDRIVE_FOLDER_MIME_TYPE:
            nested_ids = gather_all_file_and_folder_ids_recursive(service, item["id"])
            res.update(nested_ids)
        else:
            res.add(item["id"])
    return list(res)
