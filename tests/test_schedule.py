from datetime import datetime, timezone

from access_fixtures import AccessFixture, flight_row
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.lookup import build_lookup
from aeroroute.catalog.services import ResolutionPolicy, ServiceCatalog, build_services
from aeroroute.catalog.schedule import ScheduleStore, build_schedule


class ScheduleTests(AccessFixture):
    def schedule(self, rows=None, name="dataset"):
        snapshot = self.snapshot(rows, name=name)
        store = DatasetStore(snapshot, build_lookup(snapshot, self.root / "lookups"))
        self.addCleanup(store.close)
        catalog = ServiceCatalog(store, build_services(store, self.root / "services",
                                 ResolutionPolicy(review_reference="synthetic-review")))
        self.addCleanup(catalog.close)
        periods = sorted({(p["source"]["year"], p["source"]["month"]) for p in store.snapshot["partitions"]})
        schedule = ScheduleStore(catalog, build_schedule(catalog, self.root / "schedules", periods, shard_rows=2))
        self.addCleanup(schedule.close)
        return schedule

    def test_query_bounds_and_service_exclusion(self):
        schedule = self.schedule()
        start = datetime(2020, 1, 1, 16, tzinfo=timezone.utc)
        end = datetime(2020, 1, 2, 16, tzinfo=timezone.utc)
        result = schedule.departures(14771, start, end)
        self.assertTrue(result.complete)
        self.assertEqual(len(result.flights), 1)
        flight = result.flights[0]
        self.assertEqual(flight.scheduled_departure_at, start)
        self.assertEqual(schedule.get_flight(flight.flight_id), flight)
        self.assertEqual(schedule.departures(14771, start, end, (flight.flight_id,)).flights, ())
        self.assertEqual(flight.marketing_flight_number, "0012")

    def test_missing_month_is_not_empty_schedule(self):
        schedule = self.schedule()
        with self.assertRaisesRegex(ValueError, "coverage"):
            schedule.departures(14771, datetime(2020, 2, 5, tzinfo=timezone.utc),
                                datetime(2020, 2, 6, tzinfo=timezone.utc))

    def test_schedule_equivalence_allows_provenance_changes(self):
        ordinary = self.schedule([flight_row()], "ordinary")
        cancelled = self.schedule([flight_row(Cancelled="1", DepTime="", ArrTime="", DepDelay="", ArrDelay="")], "cancelled")
        left, right = list(ordinary.all_flights()), list(cancelled.all_flights())
        def projection(flight):
            return {key: value for key, value in flight.to_dict().items() if key not in ("schedule_source", "schedule_ref")}
        self.assertEqual([projection(f) for f in left], [projection(f) for f in right])
        self.assertNotEqual(left[0].schedule_source.source_sha256, right[0].schedule_source.source_sha256)
        self.assertNotEqual(left[0].schedule_ref, right[0].schedule_ref)
        self.assertNotIn("cancelled", left[0].to_dict())

    def test_query_crosses_utc_year_boundary_using_local_month_coverage(self):
        schedule = self.schedule([flight_row("2020-12-31", CRSDepTime="2300", CRSArrTime="0700"),
                                  flight_row("2021-01-01", CRSDepTime="0100", CRSArrTime="0900")])
        result = schedule.departures(14771, datetime(2021, 1, 1, 7, tzinfo=timezone.utc),
                                     datetime(2021, 1, 1, 10, tzinfo=timezone.utc))
        self.assertEqual(len(result.flights), 2)
        self.assertEqual([flight.scheduled_departure_at.hour for flight in result.flights], [7, 9])

    def test_pagination_is_explicit_complete_and_bound_to_query(self):
        schedule = self.schedule([flight_row(f"2020-01-{day:02d}") for day in range(1, 5)])
        start, end = datetime(2020, 1, 1, 16, tzinfo=timezone.utc), datetime(2020, 1, 6, 16, tzinfo=timezone.utc)
        first = schedule.departures(14771, start, end, page_size=2)
        self.assertFalse(first.complete)
        self.assertIsNotNone(first.next_cursor)
        second = schedule.departures(14771, start, end, page_size=2, cursor=first.next_cursor)
        self.assertTrue(second.complete)
        self.assertIsNone(second.next_cursor)
        self.assertEqual(len({flight.flight_id for flight in first.flights + second.flights}), 4)
        with self.assertRaises(ValueError):
            schedule.departures(14771, start, end, (first.flights[0].flight_id,), cursor=first.next_cursor)

    def test_naive_or_empty_query_is_rejected(self):
        schedule = self.schedule()
        with self.assertRaises(ValueError):
            schedule.departures(14771, datetime(2020, 1, 1), datetime(2020, 1, 2))
        instant = datetime(2020, 1, 1, 16, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            schedule.departures(14771, instant, instant)

    def test_duplicate_actions_cannot_be_retried_under_an_alias(self):
        schedule = self.schedule([flight_row(), flight_row(Marketing_Airline_Network="XX")])
        flights = list(schedule.all_flights())
        self.assertEqual(len(flights), 1)
        self.assertEqual(len(schedule.catalog.supports(flights[0].flight_id)), 2)
