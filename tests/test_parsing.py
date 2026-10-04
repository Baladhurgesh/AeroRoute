import unittest

from aeroroute.domain.scalars import parse_integer
from aeroroute.normalization.parsing import parse_boolean


class ParsingTests(unittest.TestCase):
    def test_numeric_parsing_is_tolerant_but_not_truncating(self):
        for raw, expected in (("15.00", 15), ("-15.0", -15), ("15.7", None), ("", None), ("NaN", None),
                              ("nonsense", None), ("1e9999", None)):
            issues = []
            with self.subTest(raw=raw):
                self.assertEqual(parse_integer(raw, "duration", issues), expected)
                if raw and expected is None:
                    self.assertEqual(issues[0]["raw_value"], raw)
        self.assertIsNone(parse_boolean("2", "cancelled", []))
        self.assertFalse(parse_boolean("0.00", "cancelled", []))
