import unittest
from datetime import date, datetime, timedelta, timezone

from aeroroute.normalization.times import combine_arrivals, schedule_times, diversion_sequence
from aeroroute.reference.timezones import pinned_zone


UTC = timezone.utc
T = datetime(2020, 1, 1, 12, tzinfo=UTC)


class ReconstructionTests(unittest.TestCase):
    def test_arrival_paths_independent(self):
        for elapsed, delay, method in ((T, None, "elapsed_only"), (None, T, "arrival_delay_only"),
                                        (T, T, "both_agree")):
            with self.subTest(method=method):
                fact = combine_arrivals(elapsed, delay)
                self.assertEqual(fact.value, T)
                self.assertEqual(fact.method, method)
                self.assertEqual(fact.status, "resolved")
        self.assertIsNone(combine_arrivals(None, None).value)

    def test_disagreeing_arrival_paths_preserve_candidates(self):
        later = T + timedelta(minutes=2)
        fact = combine_arrivals(T, later)
        self.assertEqual(fact.status, "contradiction")
        self.assertIsNone(fact.value)
        self.assertEqual(fact.evidence, (T, later))
        self.assertEqual(combine_arrivals(T, later, tolerance_minutes=2).value, T)

    def test_overnight_schedule(self):
        dep, arr = schedule_times(date(2020, 1, 1), "2300", "0700", 300,
                                  pinned_zone("America/Los_Angeles"), pinned_zone("America/New_York"))
        self.assertEqual(dep.value, datetime(2020, 1, 2, 7, tzinfo=UTC))
        self.assertEqual(arr.value, datetime(2020, 1, 2, 12, tzinfo=UTC))

    def test_missing_schedule_crosscheck_does_not_erase_arrival(self):
        dep, arr = schedule_times(date(2020, 1, 1), "0800", "", 300,
                                  pinned_zone("America/Los_Angeles"), None)
        self.assertIsNotNone(dep.value)
        self.assertEqual(arr.value, dep.value + timedelta(minutes=300))
        self.assertIn("reported_clock_missing", arr.checks)

    def test_schedule_contradiction_does_not_erase_departure(self):
        dep, arr = schedule_times(date(2020, 1, 1), "0800", "0900", 300,
                                  pinned_zone("America/Los_Angeles"), pinned_zone("America/New_York"))
        self.assertIsNotNone(dep.value)
        self.assertEqual(arr.status, "contradiction")

    def test_dst_gap_fold_and_midnight_remain_explicit(self):
        zone = pinned_zone("America/New_York")
        for day, clock, issue in ((date(2020, 3, 8), "0230", "nonexistent_local_time"),
                                  (date(2020, 11, 1), "0130", "ambiguous_local_time"),
                                  (date(2020, 1, 1), "2400", "midnight_date_ambiguous")):
            with self.subTest(day=day, clock=clock):
                dep, arr = schedule_times(day, clock, None, None, zone, None)
                self.assertIsNone(dep.value)
                self.assertIn(issue, dep.issues)

    def test_schedule_only_evidence_can_resolve_fold(self):
        dep, arr = schedule_times(date(2020, 11, 1), "0130", "0230", 120,
                                  pinned_zone("America/New_York"), pinned_zone("America/New_York"))
        self.assertEqual(dep.value, datetime(2020, 11, 1, 5, 30, tzinfo=UTC))
        self.assertEqual(arr.value, datetime(2020, 11, 1, 7, 30, tzinfo=UTC))

    def test_diversion_requires_upper_anchor(self):
        result = diversion_sequence([("1300", "1400", pinned_zone("UTC"))], T, None)
        self.assertIsNone(result[0][0].value)
        self.assertIn("missing_time_anchor", result[0][0].issues)

    def test_diversion_sequence_preserves_unknown_intermediate_time(self):
        result = diversion_sequence([("1300", "", pinned_zone("UTC")),
                                     ("1500", "1600", pinned_zone("UTC"))], T, T + timedelta(hours=5))
        self.assertEqual(result[0][0].value, T + timedelta(hours=1))
        self.assertIsNone(result[0][1].value)
        self.assertEqual(result[1][0].value, T + timedelta(hours=3))

    def test_diversion_multiple_possible_dates_not_guessed(self):
        result = diversion_sequence([("1300", "", pinned_zone("UTC"))], T, T + timedelta(days=2))
        self.assertIsNone(result[0][0].value)
        self.assertIn("ambiguous_stop_date", result[0][0].issues)
