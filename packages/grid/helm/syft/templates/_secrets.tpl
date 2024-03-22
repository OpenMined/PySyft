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
Set a value for a Secret.
- If the secret exists, the existing value will be re-used.
- If "randomDefault"=true, a random value will be generated.
- If "randomDefault"=false, the "default" value will be used.

Usage:
  Generate random secret of length 64
  {{- include "common.secrets.set " (dict "secret" "some-secret-name" "randomDefault" true "randomLength" 64 "context" $ ) }}

  Use a static default value (with random disabled)
  {{- include "common.secrets.set " (dict "secret" "some-secret-name" "default" "default-value" "randomDefault" false "context" $ ) }}

Params:
  secret - String (Required) - Name of the 'Secret' resource where the key is stored.
  key - String - (Required) - Name of the key in the secret.
  randomDefault - Bool - (Optional) - If true, a random value will be generated if secret does note exit.
  randomLength - Int - (Optional) - The length of the generated secret. Default is 32.
  default - String - (Optional) - Default value to use if the secret does not exist if "randomDefault" is set to false.
  context - Context (Required) - Parent context.
*/}}
{{- define "common.secrets.set" -}}
  {{- $secretVal := "" -}}
  {{- $existingSecret := include "common.secrets.get" (dict "secret" .secret "key" .key "context" .context ) | default "" -}}

  {{- if $existingSecret -}}
    {{- $secretVal = $existingSecret -}}
  {{- else if .randomDefault -}}
    {{- $length := .randomLength | default 32 -}}
    {{- $secretVal = randAlphaNum $length | b64enc -}}
  {{- else -}}
    {{- $secretVal = .default | required (printf "default value required for secret=%s key=%s" .secret .key) |b64enc -}}
  {{- end -}}

  {{- printf "%s" $secretVal -}}

{{- end -}}
