import unittest
from pathlib import Path

import app


class AppTests(unittest.TestCase):
    def test_load_incidents_reads_rows(self) -> None:
        incidents = app.load_incidents(Path("California_Fire_Incidents.csv"), limit=3)
        self.assertEqual(len(incidents), 3)
        self.assertIn("Name", incidents[0])

    def test_compute_summary_has_expected_keys(self) -> None:
        incidents = [
            {
                "AcresBurned": "10",
                "StructuresDestroyed": "2",
                "Active": "TRUE",
                "Counties": "A, B",
                "Started": "2020-01-02",
            },
            {
                "AcresBurned": "5.5",
                "StructuresDestroyed": "0",
                "Active": "FALSE",
                "Counties": "B",
                "Started": "2020-01-03",
            },
        ]
        summary = app.compute_summary(incidents)
        self.assertEqual(summary["totalIncidents"], 2)
        self.assertEqual(summary["activeIncidents"], 1)
        self.assertEqual(summary["countiesImpacted"], 2)
        self.assertEqual(summary["latestIncidentDate"], "2020-01-03")


if __name__ == "__main__":
    unittest.main()
