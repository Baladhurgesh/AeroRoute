import unittest
from datetime import date, datetime

from aeroroute.reference.airports import AirportResolver
from aeroroute.reference.timezones import pinned_zone


class AirportTimezoneTests(unittest.TestCase):
    def test_pinned_timezones_include_hawaii_and_arizona(self):
        for key in ("Pacific/Honolulu", "America/Phoenix"):
            zone = pinned_zone(key)
            self.assertEqual(datetime(2020, 1, 1, tzinfo=zone).utcoffset(),
                             datetime(2020, 7, 1, tzinfo=zone).utcoffset())

    def test_unknown_airport_not_guessed(self):
        result = AirportResolver().resolve(999999, "ZZZ", date(2020, 1, 1))
        self.assertIsNone(result["time_zone"])
        self.assertEqual(result["status"], "unmapped")
