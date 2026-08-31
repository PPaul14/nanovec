# nanovec -- multi-stage build.
#
# Two stages so the C compiler and pip's caches stay in the builder and never
# reach the final image: smaller to pull, and no toolchain sitting in a running
# service for an attacker to use.

# --------------------------------------------------------------------------- #
# Stage 1: builder
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

# NumPy normally installs from a prebuilt wheel. build-essential is here only
# for the case where no wheel matches the target platform and it has to compile
# from source -- on arm64, say. It stays in this stage and is discarded.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than the system site-packages, because it gives the
# runtime stage exactly one self-contained directory to copy.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy the dependency manifest ALONE, before any source. Docker caches layers
# by instruction, so this layer is only invalidated when requirements.txt
# changes. Copying the whole project first would make every one-line edit to
# app/ bust the cache and reinstall NumPy from scratch.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

# Run as an unprivileged user. If the process is ever compromised, it does not
# own the container.
RUN useradd --create-home --shell /usr/sbin/nologin nanovec

# PYTHONUNBUFFERED: stdout/stderr go straight out instead of sitting in a
#   buffer, so `docker logs` shows what is happening now rather than minutes ago.
# PYTHONDONTWRITEBYTECODE: no .pyc files -- they would be written into a
#   read-mostly layer for no benefit and only add image weight.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Just the built venv -- no compiler, no apt lists, no pip cache.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/

# The snapshot and write-ahead log live here. Anything written to a container's
# own filesystem dies with the container, so without this volume the index would
# silently vanish on every `docker compose down` and Week 3's durability work
# would be worthless in practice. chown because the process is not root.
RUN mkdir -p /app/data && chown -R nanovec:nanovec /app
VOLUME ["/app/data"]

USER nanovec

EXPOSE 8000

# /stats reads the live index, so a 200 here proves the index actually recovered
# from disk -- not merely that a socket is accepting connections. urllib is in
# the standard library; adding curl to the image just for a healthcheck would
# grow it for nothing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/stats', timeout=4).status == 200 else sys.exit(1)"

# 0.0.0.0, not 127.0.0.1: inside a container, 127.0.0.1 means "this container
# only", and a published port would never reach the process.
#
# --workers 1 is a CORRECTNESS REQUIREMENT, not a performance default. The index
# is plain unsynchronised Python objects held in process memory. A second worker
# is a separate OS process with its own divergent copy of the graph: writes to
# one are invisible to the other, and each would snapshot over the other's state.
# Raising this number does not scale nanovec, it corrupts it. Concurrency would
# need either a lock and shared memory, or a single writer process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
