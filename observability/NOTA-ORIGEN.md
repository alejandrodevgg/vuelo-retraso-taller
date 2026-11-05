# NOTA DE ORIGEN — carpeta observability/

Ficheros copiados del material del curso **"OpenTelemetry in Practice" (Packt,
ISBN 9781807783334)**, carpeta `compose/`, para usarlos como stack de
observabilidad del taller MLOps (sección s9, vía `include:` en
`estados-finalizados/v0.7-observability.yaml`).

- `compose.observability.yaml` — **verbatim** (pins intactos: Prometheus
  v3.9.1, Loki 3.6.6, Tempo 2.10.1, Grafana 12.3, OTel Collector Contrib
  0.146.1; volúmenes persistentes y `depends_on` incluidos).
- `otel-collector-config.yaml` — **verbatim** (OTLP gRPC 4317 / HTTP 4318,
  pipelines traces→Tempo, metrics→Prometheus-exporter, logs→Loki;
  `memory_limiter` 512 MiB).
- `loki-config.yaml`, `tempo-config.yaml`, `grafana-datasources.yaml` —
  **verbatim** (los datasources de Grafana vienen ya aprovisionados con la
  correlación log→trace hacia Tempo: eso es oro del curso y no se toca).
- `prometheus.yaml` — **una diferencia documentada**: el job `redis` del demo
  de traducción del curso se sustituye por `vuelo-retraso-api` (scrapea
  `api:8000/metrics`). Todo lo demás igual.

Regla del taller: **no subir pins de OTel/Grafana a mitad de curso** — el
service map y la correlación trazas↔logs dependen del cableado exacto de estos
ficheros.
