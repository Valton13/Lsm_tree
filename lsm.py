"""
A toy LSM tree, built for learning.  Needs Python 3.10+.

Write path : put/delete -> WAL -> memtable -> flush -> new SSTable in L0
Read path  : memtable -> L0 (newest first) -> at most ONE candidate file per level L1, L2, ...
Scan       : k-way merge of the memtable and every overlapping SSTable
Compaction : leveled.
               L0 -> L1 when L0 has `l0_trigger` files (L0 files overlap each other)
               Ln -> Ln+1 when level n holds more than its entry budget
             Newest version wins.  Tombstones are dropped once nothing deeper could
             still hold an older version of the key.
MANIFEST   : JSON file listing which SSTables are live, and in which level.  It is
             replaced atomically, so a compaction swaps its files all-or-nothing.
             Files that are not in the MANIFEST are leftovers from a crash and are
             deleted at startup.

Toy rules: keys and values are strings (None is the tombstone), single-threaded,
don't write while you are iterating a scan(), sizes are counted in entries not bytes.
"""

import bisect
import hashlib
import heapq
import itertools
import json
import math
import os
from collections import Counter

MISSING = object()  # "this table has no entry for the key" (None means "deleted")


# --------------------------------------------------------------------------
# Bloom filter
# --------------------------------------------------------------------------
class BloomFilter:
    def __init__(self, expected_keys, bits_per_key=10):
        self.m = max(8, expected_keys * bits_per_key)
        self.k = max(1, round(bits_per_key * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, key):
        d = hashlib.blake2b(key.encode(), digest_size=16).digest()
        h1 = int.from_bytes(d[:8], "little")
        h2 = int.from_bytes(d[8:], "little") | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]  # double hashing

    def add(self, key):
        for p in self._positions(key):
            self.bits[p >> 3] |= 1 << (p & 7)

    def might_contain(self, key):
        return all((self.bits[p >> 3] >> (p & 7)) & 1 for p in self._positions(key))


# --------------------------------------------------------------------------
# SSTable: an immutable, sorted file
# --------------------------------------------------------------------------
class SSTable:
    INDEX_EVERY = 16  # sparse index: remember the byte offset of every 16th key

    def __init__(self, path, stats, use_bloom=True):
        self.path = path
        self.name = os.path.basename(path)
        self.stats = stats
        self.index_keys, self.index_offsets = [], []

        keys, offset = [], 0
        with open(path, "rb") as f:
            for i, line in enumerate(f):
                key, _ = json.loads(line)
                if i % self.INDEX_EVERY == 0:
                    self.index_keys.append(key)
                    self.index_offsets.append(offset)
                keys.append(key)
                offset += len(line)
        assert keys, f"{path} is empty; the tree never writes empty tables"
        self.count = len(keys)
        self.min_key, self.max_key = keys[0], keys[-1]

        self.bloom = None
        if use_bloom:
            self.bloom = BloomFilter(len(keys))
            for k in keys:
                self.bloom.add(k)

    @staticmethod
    def write(path, sorted_items):
        """Write sorted (key, value) pairs to `path`; readers never see a partial file."""
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.writelines(
                json.dumps([key, value]).encode() + b"\n" for key, value in sorted_items
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def overlaps(self, lo=None, hi=None):
        """Does this table's key range intersect [lo, hi]?  None = unbounded."""
        return (hi is None or self.min_key <= hi) and (lo is None or self.max_key >= lo)

    def get(self, key):
        self.stats["table_lookups"] += 1
        if key < self.min_key or key > self.max_key:
            self.stats["range_skips"] += 1
            return MISSING
        if self.bloom is not None and not self.bloom.might_contain(key):
            self.stats["bloom_skips"] += 1
            return MISSING
        self.stats["disk_reads"] += 1
        i = bisect.bisect_right(self.index_keys, key) - 1  # >= 0 because key >= min_key
        with open(self.path, "rb") as f:
            f.seek(self.index_offsets[i])
            for _ in range(self.INDEX_EVERY):  # scan one index block only
                line = f.readline()
                if not line:
                    break
                k, v = json.loads(line)
                if k == key:
                    return v
                if k > key:
                    break
        return MISSING

    def items(self):
        with open(self.path, "rb") as f:
            for line in f:
                k, v = json.loads(line)
                yield k, v

    def scan(self, start=None, end=None):
        """Yield (key, value) with start <= key < end, in order."""
        i = (
            0
            if start is None
            else max(bisect.bisect_right(self.index_keys, start) - 1, 0)
        )
        with open(self.path, "rb") as f:
            f.seek(self.index_offsets[i])  # jump close to `start`
            for line in f:
                k, v = json.loads(line)
                if start is not None and k < start:
                    continue
                if end is not None and k >= end:
                    return
                yield k, v


def _tag(stream, rank):
    """Attach an age rank to every entry (0 = newest) so merges can break ties."""
    for k, v in stream:
        yield k, rank, v


# --------------------------------------------------------------------------
# The LSM tree
# --------------------------------------------------------------------------
class LSMTree:
    def __init__(
        self,
        directory,
        memtable_limit=1000,
        l0_trigger=4,
        target_file_entries=1000,
        l1_max_entries=4000,
        fanout=10,
        use_bloom=True,
        sync_wal=False,
    ):
        self.dir = directory
        self.memtable_limit = memtable_limit  # entries before a flush
        self.l0_trigger = l0_trigger  # L0 files that trigger L0 -> L1
        self.target_file_entries = (
            target_file_entries  # max entries per compaction output file
        )
        self.l1_max_entries = l1_max_entries  # L1 budget; Ln = this * fanout^(n-1)
        self.fanout = fanout
        self.use_bloom = use_bloom
        self.sync_wal = sync_wal

        self.stats = Counter()
        self.pointers = {}  # round-robin compaction cursor per level (in memory only)
        self.levels = [[]]  # levels[0]: oldest -> newest, files may overlap
        # levels[n>=1]: sorted by min_key, files never overlap
        self.next_id = 0
        self.memtable = {}

        os.makedirs(directory, exist_ok=True)
        self._manifest_path = os.path.join(directory, "MANIFEST")
        self._load_manifest()
        self._wal_path = os.path.join(directory, "wal.log")
        self._replay_wal()
        self._wal = open(self._wal_path, "a", encoding="utf-8")

    # ---- MANIFEST --------------------------------------------------------
    def _load_manifest(self):
        live = set()
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            self.next_id = data["next_id"]
            self.levels = [
                [
                    SSTable(os.path.join(self.dir, n), self.stats, self.use_bloom)
                    for n in names
                ]
                for names in data["levels"]
            ] or [[]]
            live = {n for names in data["levels"] for n in names}
        for name in os.listdir(self.dir):  # crash leftovers: not in the MANIFEST
            if name.endswith(".tmp") or (name.endswith(".sst") and name not in live):
                os.remove(os.path.join(self.dir, name))

    def _commit(self):
        """Atomically publish the current set of live files."""
        data = {
            "next_id": self.next_id,
            "levels": [[t.name for t in lvl] for lvl in self.levels],
        }
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._manifest_path)  # the single atomic switch-over point
        try:  # make the rename itself durable (POSIX)
            fd = os.open(self.dir, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass

    def _alloc_path(self):
        path = os.path.join(self.dir, f"{self.next_id:06d}.sst")
        self.next_id += 1
        return path

    def _open_table(self, path):
        return SSTable(path, self.stats, self.use_bloom)

    # ---- WAL -------------------------------------------------------------
    def _replay_wal(self):
        if not os.path.exists(self._wal_path):
            return
        with open(self._wal_path, encoding="utf-8") as f:
            for line in f:
                try:
                    key, value = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn final write
                self.memtable[key] = value

    def _log(self, key, value):
        self._wal.write(json.dumps([key, value]) + "\n")
        self._wal.flush()
        if self.sync_wal:
            os.fsync(self._wal.fileno())

    # ---- write path ------------------------------------------------------
    def put(self, key, value):
        assert isinstance(key, str) and isinstance(value, str)
        self._write(key, value)

    def delete(self, key):
        self._write(key, None)  # None = tombstone

    def _write(self, key, value):
        self.stats["user_writes"] += 1
        self._log(key, value)  # 1. WAL
        self.memtable[key] = value  # 2. memtable
        if len(self.memtable) >= self.memtable_limit:
            self.flush()  # 3. flush when full

    def flush(self):
        if not self.memtable:
            return
        path = self._alloc_path()
        items = sorted(self.memtable.items())
        SSTable.write(path, items)
        self.stats["entries_flushed"] += len(items)
        self.levels[0].append(self._open_table(path))
        self._commit()  # file is "live" only after this
        self.memtable.clear()
        self._wal.close()  # (a crash before this line just
        os.remove(self._wal_path)  #  replays duplicates: harmless)
        self._wal = open(self._wal_path, "a", encoding="utf-8")
        self._maybe_compact()

    # ---- read path -------------------------------------------------------
    def get(self, key):
        if key in self.memtable:
            return self.memtable[key]  # may be None (deleted)
        for table in reversed(self.levels[0]):  # L0: any file may hold it, newest first
            result = table.get(key)
            if result is not MISSING:
                return result
        for level in self.levels[1:]:  # L1+: at most one file per level
            table = self._find_file(level, key)
            if table is not None:
                result = table.get(key)
                if result is not MISSING:
                    return result
        return None

    @staticmethod
    def _find_file(level, key):
        i = bisect.bisect_right(level, key, key=lambda t: t.min_key) - 1
        if i >= 0 and level[i].max_key >= key:
            return level[i]
        return None

    # ---- range scan ------------------------------------------------------
    def scan(self, start=None, end=None):
        """Yield live (key, value) pairs with start <= key < end, sorted by key."""
        mem = sorted(
            (k, v)
            for k, v in self.memtable.items()
            if (start is None or k >= start) and (end is None or k < end)
        )
        streams = [_tag(mem, 0)]  # rank 0: the memtable is newest
        rank = 1
        for table in reversed(self.levels[0]):  # then L0, newest first
            if table.overlaps(start, end):
                streams.append(_tag(table.scan(start, end), rank))
                rank += 1
        for level in self.levels[1:]:  # then each deeper level
            files = [t for t in level if t.overlaps(start, end)]
            if files:
                chain = itertools.chain.from_iterable(t.scan(start, end) for t in files)
                streams.append(_tag(chain, rank))  # a level is one sorted run
                rank += 1
        return self._newest_wins(heapq.merge(*streams))

    @staticmethod
    def _newest_wins(merged):
        last = MISSING
        for key, _rank, value in merged:  # sorted by (key, rank)
            if key == last:
                continue  # older version, shadowed
            last = key
            if value is not None:  # hide tombstones from the caller
                yield key, value

    # ---- leveled compaction ---------------------------------------------
    def _level_limit(self, n):
        return self.l1_max_entries * self.fanout ** (n - 1)

    def _level_entries(self, n):
        return sum(t.count for t in self.levels[n])

    def _maybe_compact(self):
        while True:
            if len(self.levels[0]) >= self.l0_trigger:
                self._compact_level(0)
                continue
            for n in range(1, len(self.levels)):
                if self._level_entries(n) > self._level_limit(n):
                    self._compact_level(n)
                    break
            else:
                return  # nothing over budget

    def _pick_file(self, n):
        """Round-robin: take the first file after where we compacted last time."""
        files = self.levels[n]
        cursor = self.pointers.get(n)
        pick = next(
            (t for t in files if cursor is None or t.min_key > cursor), files[0]
        )
        self.pointers[n] = pick.max_key
        return pick

    def _compact_level(self, n):
        out = n + 1
        while len(self.levels) <= out:
            self.levels.append([])

        # 1. choose inputs from level n (all of L0, or one file of Ln) ...
        inputs = list(self.levels[0]) if n == 0 else [self._pick_file(n)]
        lo = min(t.min_key for t in inputs)
        hi = max(t.max_key for t in inputs)
        # 2. ... plus every file in level n+1 whose key range overlaps them
        lower = [t for t in self.levels[out] if t.overlaps(lo, hi)]

        # Trivial move: one input, nothing to merge with -> just re-label the file.
        if len(inputs) == 1 and not lower:
            self.levels[n].remove(inputs[0])
            self.levels[out].append(inputs[0])
            self.levels[out].sort(key=lambda t: t.min_key)
            self._commit()
            self.stats["trivial_moves"] += 1
            return

        # 3. merge.  Age order: newer L0 files, then the Ln input, then level n+1.
        ordered = list(reversed(inputs)) if n == 0 else inputs
        streams = [_tag(t.items(), r) for r, t in enumerate(ordered)]
        streams += [_tag(t.items(), len(ordered)) for t in lower]

        # A tombstone may be dropped only if no deeper level could hold an older
        # version of the key (otherwise that old value would come back to life).
        drop_tombstones = not any(
            t.overlaps(lo, hi) for deeper in self.levels[out + 1 :] for t in deeper
        )

        def merged():
            last = MISSING
            for key, _rank, value in heapq.merge(*streams):
                if key == last:
                    continue
                last = key
                if value is None and drop_tombstones:
                    continue
                yield key, value

        # 4. write the output as several bounded-size, non-overlapping files
        new_tables, stream = [], merged()
        while True:
            chunk = list(itertools.islice(stream, self.target_file_entries))
            if not chunk:
                break
            path = self._alloc_path()
            SSTable.write(path, chunk)
            self.stats["entries_compacted"] += len(chunk)
            new_tables.append(self._open_table(path))

        # 5. swap atomically: the MANIFEST rename is the commit point
        old = inputs + lower
        self.levels[n] = [t for t in self.levels[n] if t not in inputs]
        self.levels[out] = sorted(
            [t for t in self.levels[out] if t not in lower] + new_tables,
            key=lambda t: t.min_key,
        )
        self._commit()
        for t in old:  # a crash before this = orphans,
            os.remove(t.path)  # cleaned up on next startup
        self.stats["compactions"] += 1

    # ---- introspection ---------------------------------------------------
    def shape(self):
        return [
            (n, len(lvl), sum(t.count for t in lvl))
            for n, lvl in enumerate(self.levels)
        ]

    def describe(self):
        return " | ".join(f"L{n}: {f} files, {e} entries" for n, f, e in self.shape())

    def write_amplification(self):
        """Entries written to SSTables per user write (WAL not counted)."""
        user = self.stats["user_writes"]
        total = self.stats["entries_flushed"] + self.stats["entries_compacted"]
        return total / user if user else 0.0

    def check_invariants(self):
        for n, level in enumerate(self.levels):
            assert all(t.count > 0 for t in level)
            if n >= 1:
                for a, b in zip(level, level[1:]):
                    assert a.max_key < b.min_key, (
                        f"L{n} files overlap: {a.name}, {b.name}"
                    )

    def close(self):
        self._wal.close()


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import shutil
    import tempfile

    d = tempfile.mkdtemp()
    db = LSMTree(
        d,
        memtable_limit=20,
        l0_trigger=3,
        target_file_entries=15,
        l1_max_entries=45,
        fanout=3,
    )
    rng = random.Random(7)
    for i in range(400):
        db.put(f"user:{rng.randint(0, 199):04d}", f"v{i}")
    for i in range(0, 200, 10):
        db.delete(f"user:{i:04d}")
    db.flush()

    print("shape        :", db.describe())
    print("stats        :", dict(db.stats))
    print(f"write amp    : {db.write_amplification():.1f}x")
    print("scan 0100-0112:", list(db.scan("user:0100", "user:0112")))
    print("MANIFEST     :", open(os.path.join(d, "MANIFEST")).read()[:160], "...")
    db.check_invariants()
    shutil.rmtree(d)
