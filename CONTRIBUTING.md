# Contribuir

Gracias por el interés. El proyecto es pequeño a propósito; estas son las
pautas para que un cambio se pueda revisar y aceptar sin fricción.

## Antes de escribir código

Para cambios de cierto tamaño (nueva etapa del pipeline, cambio de esquema de
configuración, nueva integración externa), abre primero un issue describiendo
el problema y el enfoque. Para arreglos pequeños y evidentes, un PR directo
está bien.

## Entorno de desarrollo

```bash
uv sync
uv run pytest tests -q
```

Los tests no tocan la GPU ni ninguna carpeta real: trabajan sobre árboles de
directorios temporales (`tmp_path` de pytest) y mockean las llamadas a
OpenRouter. Si tu cambio toca `asr.py`, `diarize.py` o `audio.py`, prueba
también a mano con un audio real si puedes: esas rutas son difíciles de cubrir
por completo con dobles.

## Estilo

- El código y los comentarios están en español; los identificadores de la API
  de terceros (`speaker`, `segments`...) se mantienen en su idioma original.
- Comentarios solo donde el código no explica el porqué por sí solo (una
  decisión no obvia, una limitación de hardware, un bug de una librería). No
  comentes lo que el código ya dice.
- Nada de dependencias nuevas sin buena razón: el proyecto usa deliberadamente
  poco (`whisperx`, `pyannote` que trae consigo, y `urllib` de la librería
  estándar para OpenRouter).

## Privacidad y secretos

Nunca añadas al repositorio:
- claves de API, tokens, o cualquier credencial (van en `~/.config/aura-transcribe/env`, fuera del repo);
- rutas absolutas de tu máquina personal en el código o en los ficheros de systemd (usa `$HOME`, `%h`, o hazlas configurables);
- contenido real de transcripciones o notas (el `vault/` generado nunca se versiona: está en `.gitignore`).

## Tipos de nota y prompts

Si tu cambio afecta a `clasificar.py` o `notas.py`, recuerda que las
categorías son datos de configuración (`CategoriaNota` en `config.py`), no
código: evita reintroducir categorías o idiomas hardcodeados en el pipeline.
