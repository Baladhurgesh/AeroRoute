from access_fixtures import AccessFixture, flight_row
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.lookup import build_lookup
from aeroroute.catalog.services import ResolutionPolicy, ServiceCatalog, build_services


class ServiceTests(AccessFixture):
    def catalog(self, rows, review="synthetic-fixture-review"):
        snapshot = self.snapshot(rows)
        store = DatasetStore(snapshot, build_lookup(snapshot, self.root / "lookups"))
        self.addCleanup(store.close)
        path = build_services(store, self.root / "services", ResolutionPolicy(review_reference=review))
        catalog = ServiceCatalog(store, path)
        self.addCleanup(catalog.close)
        return catalog

    def test_supporting_rows_are_one_service(self):
        catalog = self.catalog([flight_row(), flight_row(Marketing_Airline_Network="XX")])
        services = list(catalog.services())
        self.assertEqual(len(services), 1)
        service = services[0]
        self.assertEqual(len(catalog.supports(service["service_instance_id"])), 2)
        self.assertTrue(catalog.ready("historical"))
        self.assertEqual(catalog.report["support_rows"], 2)

    def test_duplicate_y_is_not_generic_deduplication(self):
        catalog = self.catalog([flight_row(Duplicate="Y"), flight_row("2020-01-02")])
        self.assertEqual(len(list(catalog.services())), 2)
        self.assertEqual(catalog.report["duplicate_flags"]["Y"], 1)

    def test_unreviewed_policy_is_not_ready(self):
        catalog = self.catalog([flight_row()], review=None)
        self.assertFalse(catalog.ready("historical"))
        with self.assertRaises(ValueError):
            catalog.require_ready("historical")

    def test_outcome_conflict_does_not_remove_schedule_service(self):
        catalog = self.catalog([flight_row(), flight_row(Cancelled="1")])
        self.assertEqual(len(list(catalog.services())), 1)
        self.assertTrue(catalog.ready("historical", evidence=False))
        self.assertFalse(catalog.ready("historical"))
        self.assertEqual(catalog.report["evidence_conflicts"], 1)

    def test_conflicting_scheduled_arrival_blocks_readiness(self):
        catalog = self.catalog([flight_row(), flight_row(CRSElapsedTime="310", CRSArrTime="1610")])
        self.assertFalse(catalog.ready("historical", evidence=False))
        self.assertEqual(catalog.report["schedule_conflicts"], 1)

    def test_aliases_cannot_create_two_actions(self):
        catalog = self.catalog([flight_row(), flight_row(Operating_Airline="XX")])
        self.assertFalse(catalog.ready("historical", evidence=False))
        self.assertGreater(catalog.report["alias_conflicts"], 0)
