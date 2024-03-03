# Reuses the value from an existing secret, otherwise sets its value to a default value.
# Usage:
# {{ include "common.secrets.lookup" (dict "secret" "secret-name" "key" "keyName" "defaultValue" .Values.myValue) }}
# Params:
#   - secret - String - Required - Name of the 'Secret' resource where the password is stored.
#   - key - String - Required - Name of the key in the secret.
#   - defaultValue - String - Required - The path to the validating value in the values.yaml, e.g: "mysql.password". Will pick first parameter with a defined value.
#   - context - Context - Required - Parent context.
{{- define "common.secrets.lookup" -}}
{{- $value := "" -}}
{{- $secretData := (lookup "v1" "Secret" .context.Release.Namespace .secret).data -}}
{{- if and $secretData (hasKey $secretData .key) -}}
  {{- $value = index $secretData .key -}}
{{- else if .defaultValue -}}
  {{- $value = .defaultValue | toString | b64enc -}}
{{- end -}}
{{- if $value -}}
{{- printf "%s" $value -}}
{{- end -}}
{{- end -}}

# Params:
#   - devDefault - String - Required - The default value to use if devmode is enabled.
#   - length - Int - Optional - The length of the generated secret. Default is 32.
#   - context - Context - Required - Parent context.
{{- define "common.secrets.generate" -}}
{{- $value := "" -}}
{{ if .context.Values.global.devmode -}}
  {{- $value = .devDefault | toString | b64enc -}}
{{- else -}}
  {{- $length:= default 32 .length -}}
  {{- $value = randAlphaNum $length | b64enc -}}
{{- end -}}
{{- if $value -}}
{{- printf "%s" $value -}}
{{- end -}}
{{- end -}}
