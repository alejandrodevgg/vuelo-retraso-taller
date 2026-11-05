# HOWTO — Prefect 3 como orquestador local (stretch, opcional)

**Estado de verificación: smoke ephemeral REAL el 2026-11-02** (1 run
end-to-end desde dentro del contenedor, sin server Prefect; evidencia en
`docs/SMOKE-RETRAIN.md` §0 del taller hermano `mlops-prefect`). El flow vino
roto de fábrica en tres puntos que el smoke pagó: `@flow(log_level=…)` (kwarg
de Prefect 2.x → `TypeError` en 3.6), paths `house_*` obsoletos (el núcleo es
`flights` desde la conversión a clasificación) y el volumen `pipeline-data`
sin declarar en el overlay. Si aun así lo demostráis en clase, ensayadlo
primero: sigue siendo stretch.

## Qué es esto

Un único flow (`flow.py`) que envuelve las dos fases batch del pipeline:

```
datos (limpiar + contract + features)  ->  entrena (MLflow + registro)
```

con `@task(retries=1, retry_delay_seconds=10)`: el retry y el reintento son del
ORQUESTRADOR, no de tu script. Eso es lo que el `docker compose run` a mano no
te da, y es la antesala del CT (continuous training) del taller 1.

## Ejecutar

```bash
docker compose -f docker-compose.yml \
  -f estados-finalizados/prefect/compose.prefect.yaml \
  run --rm prefect-flow
```

Verás los dos tasks en el stdout con sus IDs de run y el `--register` dejando
una versión nueva en el registry (alias: cámbialo a mano en la UI y compara).

## Notas honestas

- Sin UI de Prefect: el modo ephemeral no arranca server. La lectura pendiente
  es el `prefecthq/prefect:3.6-python3.12` multi-arch con server SQLite
  self-hosted (compose oficial de Prefect 3) — 2 contenedores más.
- Ruido esperado (medido en el smoke): el worker de events de Prefect suelta
  tracebacks `sqlite3.OperationalError: database is locked` y
  `EventsWorker failed` contra la SQLite TEMPORAL del modo ephemeral. Son
  cosméticos: el run completa los dos tasks, el registro ocurre y el proceso
  sale 0. Si molestan en clase, `... 2>&1 | grep -v Traceback` y al piso de
  arriba (el taller `mlops-prefect` monta el server de verdad con UI).
- Prefect 3 mete un grafo de dependencias grande (~1-2 min de pip extra en el
  build). No lo mezcles con la corrida del Wi-Fi del aula.
- Pin: `prefect==3.9.1` (medido 2026-11-02). Tentadora la paridad con el
  server del taller hermano (3.6.29), pero el modo ephemeral de 3.6.29
  revienta contra el `fastapi==0.162.0` del requirements del pipeline: su
  `prefect.server` usa `Body(min_items=1)`, eliminado en fastapi ≥ 0.116
  (`TypeError: Body() got an unexpected keyword argument 'min_items'`). En
  ephemeral no hay server que negociar, así que el desfase de minor es
  inofensivo aquí. Ver el comentario del Dockerfile.
