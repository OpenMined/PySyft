{{/*
Lookup value from an existing secret. WILL NOT base64 decode the value.

Usage:
  {{- include "common.secrets.get" (dict "secret" "some-secret-name" "key" "keyName" "context" $) }}

Params:
  secret - String (Required) - Name of the 'Secret' resource where the key is stored.
  key - String - (Required) - Name of the key in the secret.
  context - Context (Required) - Parent context.
*/}}
{{- define "common.secrets.get" -}}
  {{- $value := "" -}}
  {{- $secretData := (lookup "v1" "Secret" .context.Release.Namespace .secret).data -}}

  {{- if and $secretData (hasKey $secretData .key) -}}
    {{- $value = index $secretData .key -}}
  {{- end -}}

  {{- if $value -}}
    {{- printf "%s" $value -}}
  {{- end -}}

{{- end -}}

{{/*
Re-use or set a new randomly generated secret value from an existing secret.
If global.useDefaultSecrets is set to true, the default value will be used if the secret does not exist.

Usage:
  {{- include "common.secrets.set " (dict "secret" "some-secret-name" "default" "default-value" "context" $ ) }}

Params:
  secret - String (Required) - Name of the 'Secret' resource where the key is stored.
  key - String - (Required) - Name of the key in the secret.
  default - String - (Optional) - Default value to use if the secret does not exist.
  length - Int - (Optional) - The length of the generated secret. Default is 32.
  context - Context (Required) - Parent context.
*/}}
{{- define "common.secrets.set" -}}
  {{- $secretVal := "" -}}
  {{- $existingSecret := include "common.secrets.get" (dict "secret" .secret "key" .key "context" .context ) | default "" -}}

  {{- if $existingSecret -}}
    {{- $secretVal = $existingSecret -}}
  {{- else if .context.Values.global.useDefaultSecrets -}}
    {{- $secretVal = .default | b64enc -}}
  {{- else -}}
    {{- $length := .length | default 32 -}}
    {{- $secretVal = randAlphaNum $length | b64enc -}}
  {{- end -}}

  {{- printf "%s" $secretVal -}}

{{- end -}}
