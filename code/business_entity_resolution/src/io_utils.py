"""TSV loading/writing. Every file is read as plain strings with no quoting and no NA parsing."""
import os

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

PARSE = pcsv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False,
                          newlines_in_values=False)


def _convert(names):
    return pcsv.ConvertOptions(column_types={c: pa.string() for c in names},
                               strings_can_be_null=False, quoted_strings_can_be_null=False)


def header(path):
    with open(path, encoding="utf-8") as f:
        return f.readline().rstrip("\n").rstrip("\r").split("\t")


def tsv_to_parquet(path, out, block_size=64 << 20):
    """Stream a TSV into a parquet file (all string columns) with bounded memory."""
    cols = header(path)
    reader = pcsv.open_csv(path, read_options=pcsv.ReadOptions(block_size=block_size),
                           parse_options=PARSE, convert_options=_convert(cols))
    writer = None
    n = 0
    for batch in reader:
        t = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(out + ".tmp", t.schema, compression="zstd")
        writer.write_table(t)
        n += t.num_rows
    writer.close()
    os.replace(out + ".tmp", out)
    return n


def write_tsv(s1_ids, id_lists, path, col):
    """One row per S1 id, even when the list is empty; ids de-duplicated in order."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(id_lists.get(s1, ())))
            f.write(s1 + "\t" + ",".join(ids) + "\n")
