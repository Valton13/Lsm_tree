"""
Benchmark the toy LSM tree.

    python bench.py            # 100,000 keys
    python bench.py 20000      # smaller, faster run

Caveat: the OS page cache keeps these small files in RAM, so the Bloom filter's
benefit here is understated.  On a cold cache or a real disk, every avoided file
open is a saved disk seek.
"""

import itertools
import random
import shutil
import sys
import tempfile
import time

from lsm import LSMTree

CFG = dict(
    memtable_limit=5000,
    l0_trigger=4,
    target_file_entries=5000,
    l1_max_entries=20000,
    fanout=10,
)


def key(i):
    return f"key{i:08d}"


def run_reads(db, keys):
    """Return (avg microseconds per lookup, stats delta)."""
    before = dict(db.stats)
    t = time.perf_counter()
    for k in keys:
        db.get(k)
    elapsed = time.perf_counter() - t
    delta = {
        name: db.stats[name] - before.get(name, 0)
        for name in ("table_lookups", "range_skips", "bloom_skips", "disk_reads")
    }
    return elapsed / len(keys) * 1e6, delta


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    rng = random.Random(42)
    workdir = tempfile.mkdtemp(prefix="lsmbench-")
    try:
        # ---- writes: random key order, so compactions have real merging to do
        order = list(range(n))
        rng.shuffle(order)
        db = LSMTree(workdir, **CFG)
        t = time.perf_counter()
        for i in order:
            db.put(key(i), f"value-{i}")
        db.flush()
        write_s = time.perf_counter() - t

        print(f"\n=== WRITES: {n:,} random-order puts ===")
        print(
            f"throughput          : {n / write_s:,.0f} writes/s  ({write_s:.1f}s total)"
        )
        print(
            f"write amplification : {db.write_amplification():.1f}x "
            f"(entries written to SSTables / user writes)"
        )
        print(
            f"compactions         : {db.stats['compactions']}  "
            f"(+{db.stats['trivial_moves']} trivial moves)"
        )
        print(f"shape               : {db.describe()}")
        db.check_invariants()

        # ---- reads
        sample = min(5000, n)
        hits = [key(i) for i in rng.sample(range(n), sample)]
        # "missing" keys sort INSIDE the key range, so min/max checks can't rule them out
        misses = [key(i) + "x" for i in rng.sample(range(n), sample)]

        rows = []
        for label, use_bloom in (("Bloom ON ", True), ("Bloom OFF", False)):
            if not use_bloom:
                db.close()
                db = LSMTree(workdir, use_bloom=False, **CFG)  # same data, no filters
            hit_us, hit_s = run_reads(db, hits)
            miss_us, miss_s = run_reads(db, misses)
            rows.append((label, hit_us, miss_us, miss_s))

        print(f"\n=== POINT READS: {sample:,} lookups each ===")
        print(
            f"{'':10}{'existing key':>16}{'missing key':>16}{'disk reads (missing)':>24}"
            f"{'bloom skips':>14}"
        )
        for label, hit_us, miss_us, miss_s in rows:
            print(
                f"{label:10}{hit_us:>13.1f} us{miss_us:>13.1f} us"
                f"{miss_s['disk_reads']:>24,}{miss_s['bloom_skips']:>14,}"
            )

        # ---- range scans
        scans, width = 50, 1000
        t = time.perf_counter()
        total = 0
        for _ in range(scans):
            start = key(rng.randint(0, max(0, n - width - 1)))
            total += len(list(itertools.islice(db.scan(start), width)))
        scan_s = time.perf_counter() - t
        print(f"\n=== RANGE SCANS: {scans} scans of {width:,} keys ===")
        print(
            f"{scan_s / scans * 1000:.1f} ms per scan  "
            f"({total / scan_s:,.0f} keys/s), merging memtable + all levels"
        )
        db.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
