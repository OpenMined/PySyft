export function throwIfError(json: { fqn: string; message: string }) {
  if (json.fqn === "syft.service.response.SyftError") {
    throw Error(`SyftError: ${json.message || "Unknown error"}`)
  }
}

export function getErrorMessage(err: unknown) {
  if (err instanceof Error) {
    return err.message
  }
  return "Unknown error"
}
