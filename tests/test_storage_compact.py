from access_fixtures import AccessFixture, flight_row
from aeroroute.catalog.services import ResolutionPolicy, ServiceCatalog, build_services
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.identity import load_json
from aeroroute.storage.lookup import build_lookup


class CompactStorageTests(AccessFixture):
    def test_compact_artifacts_preserve_service_and_source_contracts(self):
        rows = [flight_row(Flight_Number_Operating_Airline=f"{index:04d}",
                           Flight_Number_Marketing_Airline=f"{index:04d}") for index in range(1000)]
        snapshot = self.snapshot(rows, batch_size=64)
        lookup = build_lookup(snapshot, self.root / "lookups")
        self.assertEqual(load_json(lookup / "manifest.json")["identity"]["lookup_version"], 2)
        self.assertLess((lookup / "lookup.sqlite").stat().st_size, 300000)
        with DatasetStore(snapshot, lookup) as store:
            refs = tuple(store.source_refs("historical"))
            self.assertEqual(len(refs), 1000)
            self.assertEqual(store.get(refs[-1]).source.source_row_id, 1000)
            path = build_services(store, self.root / "services", ResolutionPolicy(review_reference="synthetic-review"))
            self.assertEqual(load_json(path / "manifest.json")["identity"]["builder_version"], 2)
            self.assertLess((path / "services.sqlite").stat().st_size, 1300000)
            with ServiceCatalog(store, path) as catalog:
                self.assertTrue(catalog.ready("historical"))
                services = list(catalog.services(schedule_order=True))
                self.assertEqual(len(services), 1000)
                self.assertEqual(len(catalog.supports(services[0]["service_id"])), 1)
                ref = store.source_ref(services[0]["primary_source"], "historical")
                self.assertEqual(store.get(ref).flight["operating_flight_number"], services[0]["attributes"]["operating_flight_number"])
