{{- define "common.resources.preset" -}}
{{- $presets := dict
  "nano" (dict
      "requests" (dict "cpu" "100m" "memory" "128Mi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "200m" "memory" "256Mi" "ephemeral-storage" "1Gi")
   )
  "micro" (dict
      "requests" (dict "cpu" "250m" "memory" "256Mi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "500m" "memory" "512Mi" "ephemeral-storage" "1Gi")
   )
  "small" (dict
      "requests" (dict "cpu" "500m" "memory" "512Mi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "1.0" "memory" "1Gi" "ephemeral-storage" "1Gi")
   )
  "medium" (dict
      "requests" (dict "cpu" "500m" "memory" "1Gi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "1.0" "memory" "2Gi" "ephemeral-storage" "1Gi")
   )
  "large" (dict
      "requests" (dict "cpu" "1.0" "memory" "2Gi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "2.0" "memory" "4Gi" "ephemeral-storage" "1Gi")
   )
  "xlarge" (dict
      "requests" (dict "cpu" "2.0" "memory" "4Gi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "4.0" "memory" "8Gi" "ephemeral-storage" "1Gi")
   )
  "2xlarge" (dict
      "requests" (dict "cpu" "4.0" "memory" "8Gi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "8.0" "memory" "16Gi" "ephemeral-storage" "1Gi")
   )
  "4xlarge" (dict
      "requests" (dict "cpu" "8.0" "memory" "16Gi" "ephemeral-storage" "50Mi")
      "limits" (dict "cpu" "16.0" "memory" "32Gi" "ephemeral-storage" "1Gi")
   )
 }}
{{- if hasKey $presets .type -}}
{{- index $presets .type | toYaml -}}
{{- else -}}
{{- printf "ERROR: Preset key '%s' invalid. Allowed values are %s" .type (join "," (keys $presets)) | fail -}}
{{- end -}}
{{- end -}}
