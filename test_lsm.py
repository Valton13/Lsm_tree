"""Run with:  python -m unittest -v test_lsm"""

import os
import random
import shutil
import tempfile
import unittest

from lsm import LSMTree

SMALL = dict(
    memtable_limit=20, l0_trigger=3, target_file_entries=15, l1_max_entries=45, fanout=3
)


def key(i):
    return f"k{i:04d}"


class LSMTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lsmtest-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def open(self, **overrides):
        return LSMTree(self.dir, **{**SMALL, **overrides})

    def assert_matches(self, db, ref, universe=500):
        for i in range(universe):
            self.assertEqual(db.get(key(i)), ref.get(key(i)), key(i))
        self.assertEqual(list(db.scan()), sorted(ref.items()))

    def test_random_ops_match_a_dict(self):
        rng = random.Random(1)
        db, ref = self.open(), {}
        for step in range(6000):
            k = key(rng.randint(0, 499))
            if rng.random() < 0.25:
                db.delete(k)
                ref.pop(k, None)
            else:
                v = f"v{step}"
                db.put(k, v)
                ref[k] = v
            if step % 750 == 0:
                db.check_invariants()
                self.assert_matches(db, ref)
        db.check_invariants()
        self.assertGreaterEqual(len(db.levels), 3, "test should reach L2+")
        self.assertGreater(db.stats["compactions"], 5)
        self.assert_matches(db, ref)

        for _ in range(50):  # random range scans
            a, b = sorted(rng.sample(range(500), 2))
            expected = [(k, v) for k, v in sorted(ref.items()) if key(a) <= k < key(b)]
            self.assertEqual(list(db.scan(key(a), key(b))), expected)

        db.close()
        reopened = self.open()  # state survives a restart
        reopened.check_invariants()
        self.assert_matches(reopened, ref)

    def test_wal_recovers_unflushed_writes(self):
        db = self.open(memtable_limit=1000)
        db.put("a", "1")
        db.put("b", "2")
        db.delete("a")
        # no flush, no close: simulate a crash
        db2 = self.open(memtable_limit=1000)
        self.assertIsNone(db2.get("a"))
        self.assertEqual(db2.get("b"), "2")

    def test_orphan_files_are_removed_on_startup(self):
        db = self.open()
        for i in range(100):
            db.put(key(i), "x")
        db.close()
        with open(os.path.join(self.dir, "999999.sst"), "w") as f:
            f.write('["zzz", "orphan"]\n')  # a compaction output that never committed
        with open(os.path.join(self.dir, "000123.sst.tmp"), "w") as f:
            f.write("garbage")
        db2 = self.open()
        self.assertIsNone(db2.get("zzz"))
        names = os.listdir(self.dir)
        self.assertNotIn("999999.sst", names)
        self.assertFalse(any(n.endswith(".tmp") for n in names))

    def test_delete_never_resurrects_old_values(self):
        db = self.open()
        db.put("victim", "old")
        for i in range(300):  # push "victim" down to deep levels
            db.put(key(i), "filler")
        db.delete("victim")
        for i in range(300, 900):  # keep compacting the tombstone down
            db.put(key(i), "filler")
        self.assertIsNone(db.get("victim"))
        self.assertNotIn("victim", [k for k, _ in db.scan()])

    def test_bloom_filter_does_not_change_results(self):
        rng = random.Random(3)
        db, ref = self.open(use_bloom=True), {}
        for step in range(2000):
            k = key(rng.randint(0, 499))
            db.put(k, f"v{step}")
            ref[k] = f"v{step}"
        db.close()
        no_bloom = self.open(use_bloom=False)
        self.assert_matches(no_bloom, ref)

    def test_scan_bounds(self):
        db = self.open()
        for i in range(100):
            db.put(key(i), str(i))
        self.assertEqual(
            [k for k, _ in db.scan(key(10), key(13))], [key(10), key(11), key(12)]
        )
        self.assertEqual(len(list(db.scan(start=key(95)))), 5)
        self.assertEqual(len(list(db.scan(end=key(5)))), 5)
        self.assertEqual(list(db.scan("zzz", None)), [])


if __name__ == "__main__":
    unittest.main()
