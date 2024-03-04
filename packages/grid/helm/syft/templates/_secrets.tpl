{{/*
Reuses the value from an existing secret, otherwise sets its value to a default value.

Usage:
  {{- include "common.secrets.lookup" (dict "secret" "secret-name" "key" "keyName" "value" .Values.myValue) }}

Params:
  secret - String (Required) - Name of the 'Secret' resource where the key is stored.
  key - String - (Required) - Name of the key in the secret.
  value - String (Required) - Value of the key in the secret.
  context - Context (Required) - Parent context.
*/}}
{{- define "common.secrets.lookup" -}}
{{- $value := "" -}}
{{- $secretData := (lookup "v1" "Secret" .context.Release.Namespace .secret).data -}}
{{- if and $secretData (hasKey $secretData .key) -}}
  {{- $value = index $secretData .key -}}
{{- else if .value -}}
  {{- $value = .value | toString | b64enc -}}
{{- end -}}
{{- if $value -}}
{{- printf "%s" $value -}}
{{- end -}}
{{- end -}}

{{/*
Generate a secret value. If global.useDefaultSecrets is true, the default value is used.

Usage:
  {{- include "common.secrets.generate " (dict "default" "my-default-password" "length" 32 "context" .) }}

Params:
  default - String (Required) - The default value to use if useDefaultSecrets=true.
  length - Int (Optional) - The length of the generated secret. Default is 32.
  context - Context (Required) - Parent context.
*/}}
{{- define "common.secrets.generate" -}}
{{- $value := "" -}}
{{ if .context.Values.global.useDefaultSecrets -}}
  {{- $value = .default | toString | b64enc -}}
{{- else -}}
  {{- $length:= default 32 .length -}}
  {{- $value = randAlphaNum $length | b64enc -}}
{{- end -}}
{{- if $value -}}
{{- printf "%s" $value -}}
{{- end -}}
{{- end -}}
