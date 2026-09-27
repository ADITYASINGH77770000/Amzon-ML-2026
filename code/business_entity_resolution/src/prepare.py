"""Stage 0 driver: raw parquet -> normalised parquet (parallel, bounded memory)."""
import json
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

import config
from normalize import prepare_columns

_TD = None


def _init():
    global _TD
    _TD = json.load(open(config.TDICT, encoding="utf-8")) if os.path.exists(config.TDICT) else {}


def _work(args):
    names, addrs = args
    return prepare_columns(names, addrs, _TD)


def normalise_file(src, dst, batch=100_000, pool=None):
    pf = pq.ParquetFile(src)
    writer = None
    for rb in pf.iter_batches(batch_size=batch * config.N_JOBS):
        d = rb.to_pydict()
        names, addrs = d["business_name"], d["business_address"]
        step = (len(names) + config.N_JOBS - 1) // config.N_JOBS
        parts = pool.map(_work, [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)])
        cols = {"entity_id": d["entity_id"], "country": d["country"]}
        for k in parts[0]:
            v = [x for p in parts for x in p[k]]
            cols[k] = v
        t = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(dst + ".tmp", t.schema, compression="zstd")
        writer.write_table(t)
    writer.close()
    os.replace(dst + ".tmp", dst)


def main(files=None):
    os.makedirs(config.NORM, exist_ok=True)
    files = files or ["train_source1", "train_source2", "train_source3",
                      "test_source1", "test_source2", "test_source3"]
    with Pool(config.N_JOBS, initializer=_init) as pool:
        for f in files:
            t = time.time()
            normalise_file(config.RAW + f + ".parquet", config.NORM + f + ".parquet", pool=pool)
            print(f, "normalised in", round(time.time() - t), "s", flush=True)
    for f in files:
        to_ipc(f)


def to_ipc(f):
    """Uncompressed Arrow IPC copy of a normalised table, for memory-mapping."""
    t = pq.read_table(config.NORM + f + ".parquet")
    with pa.OSFile(config.NORM + f + ".arrow", "wb") as sink:
        with pa.ipc.new_file(sink, t.schema) as w:
            w.write_table(t, max_chunksize=1 << 20)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
