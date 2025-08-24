{{/*
Expand the name of the chart.
*/}}
{{- define "hft-crypto-bot.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "hft-crypto-bot.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "hft-crypto-bot.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "hft-crypto-bot.labels" -}}
helm.sh/chart: {{ include "hft-crypto-bot.chart" . }}
{{ include "hft-crypto-bot.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "hft-crypto-bot.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hft-crypto-bot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "hft-crypto-bot.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "hft-crypto-bot.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate common environment variables
*/}}
{{- define "hft-crypto-bot.commonEnv" -}}
- name: ENVIRONMENT
  value: {{ .Values.global.environment | quote }}
- name: NAMESPACE
  value: {{ .Values.global.namespace | quote }}
- name: LOG_LEVEL
  value: {{ .Values.config.logging.level | quote }}
- name: LOG_FORMAT
  value: {{ .Values.config.logging.format | quote }}
- name: KAFKA_BOOTSTRAP_SERVERS
  value: "kafka:29092"
- name: REDIS_URL
  value: "redis://redis:6379"
{{- end }}

{{/*
Generate resource limits and requests
*/}}
{{- define "hft-crypto-bot.resources" -}}
{{- if .resources }}
resources:
  {{- if .resources.requests }}
  requests:
    {{- if .resources.requests.memory }}
    memory: {{ .resources.requests.memory }}
    {{- end }}
    {{- if .resources.requests.cpu }}
    cpu: {{ .resources.requests.cpu }}
    {{- end }}
  {{- end }}
  {{- if .resources.limits }}
  limits:
    {{- if .resources.limits.memory }}
    memory: {{ .resources.limits.memory }}
    {{- end }}
    {{- if .resources.limits.cpu }}
    cpu: {{ .resources.limits.cpu }}
    {{- end }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Generate security context
*/}}
{{- define "hft-crypto-bot.securityContext" -}}
securityContext:
  runAsNonRoot: {{ .Values.global.securityContext.runAsNonRoot }}
  runAsUser: {{ .Values.global.securityContext.runAsUser }}
  fsGroup: {{ .Values.global.securityContext.fsGroup }}
{{- end }}

{{/*
Generate image pull secrets
*/}}
{{- define "hft-crypto-bot.imagePullSecrets" -}}
{{- if .Values.global.pullSecrets }}
imagePullSecrets:
{{- range .Values.global.pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate common volume mounts
*/}}
{{- define "hft-crypto-bot.volumeMounts" -}}
- name: logs
  mountPath: /app/logs
- name: data
  mountPath: /app/data
{{- if .Values.global.persistence.enabled }}
- name: persistent-storage
  mountPath: /app/persistent
{{- end }}
{{- end }}

{{/*
Generate common volumes
*/}}
{{- define "hft-crypto-bot.volumes" -}}
- name: logs
  emptyDir: {}
- name: data
  emptyDir: {}
{{- if .Values.global.persistence.enabled }}
- name: persistent-storage
  persistentVolumeClaim:
    claimName: {{ include "hft-crypto-bot.fullname" . }}-pvc
{{- end }}
{{- end }}

{{/*
Generate service monitor labels
*/}}
{{- define "hft-crypto-bot.serviceMonitorLabels" -}}
app.kubernetes.io/name: {{ include "hft-crypto-bot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
prometheus.io/scrape: "true"
{{- end }}

{{/*
Generate HPA template
*/}}
{{- define "hft-crypto-bot.hpa" -}}
{{- if .autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .name }}-hpa
  namespace: {{ $.Values.global.namespace }}
  labels:
    {{- include "hft-crypto-bot.labels" $ | nindent 4 }}
    app.kubernetes.io/component: {{ .name }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ .name }}-service
  minReplicas: {{ .autoscaling.minReplicas }}
  maxReplicas: {{ .autoscaling.maxReplicas }}
  metrics:
  {{- if .autoscaling.targetCPUUtilizationPercentage }}
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{ .autoscaling.targetCPUUtilizationPercentage }}
  {{- end }}
  {{- if .autoscaling.targetMemoryUtilizationPercentage }}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {{ .autoscaling.targetMemoryUtilizationPercentage }}
  {{- end }}
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
{{- end }}
{{- end }}

{{/*
Generate network policy template
*/}}
{{- define "hft-crypto-bot.networkPolicy" -}}
{{- if .Values.networkPolicy.enabled }}
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ .name }}-network-policy
  namespace: {{ $.Values.global.namespace }}
  labels:
    {{- include "hft-crypto-bot.labels" $ | nindent 4 }}
    app.kubernetes.io/component: {{ .name }}
spec:
  podSelector:
    matchLabels:
      app: {{ .name }}-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: {{ $.Values.global.namespace }}
    ports:
    - protocol: TCP
      port: {{ .service.port }}
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53  # DNS
    - protocol: UDP
      port: 53  # DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: {{ $.Values.global.namespace }}
{{- end }}
{{- end }}

{{/*
Generate pod disruption budget template
*/}}
{{- define "hft-crypto-bot.podDisruptionBudget" -}}
{{- if .Values.podDisruptionBudget.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ .name }}-pdb
  namespace: {{ $.Values.global.namespace }}
  labels:
    {{- include "hft-crypto-bot.labels" $ | nindent 4 }}
    app.kubernetes.io/component: {{ .name }}
spec:
  {{- if .Values.podDisruptionBudget.minAvailable }}
  minAvailable: {{ .Values.podDisruptionBudget.minAvailable }}
  {{- end }}
  {{- if .Values.podDisruptionBudget.maxUnavailable }}
  maxUnavailable: {{ .Values.podDisruptionBudget.maxUnavailable }}
  {{- end }}
  selector:
    matchLabels:
      app: {{ .name }}-service
{{- end }}
{{- end }}
