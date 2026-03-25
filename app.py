import argparse
import csv
import json
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "California_Fire_Incidents.csv"
WEB_DIR = BASE_DIR / "web"


def parse_float(value: str) -> float:
    if value is None:
        return 0.0
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_incidents(csv_path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    incidents: List[Dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as dataset:
        reader = csv.DictReader(dataset)
        for row in reader:
            incidents.append(row)
            if limit is not None and len(incidents) >= limit:
                break
    return incidents


def compute_summary(incidents: List[Dict[str, str]]) -> Dict[str, str | int | float]:
    latest_date: datetime | None = None
    total_acres = 0.0
    total_structures = 0.0
    unique_counties = set()
    active_incidents = 0

    for incident in incidents:
        total_acres += parse_float(incident.get("AcresBurned", ""))
        total_structures += parse_float(incident.get("StructuresDestroyed", ""))
        if str(incident.get("Active", "")).strip().upper() == "TRUE":
            active_incidents += 1

        counties = str(incident.get("Counties", "")).split(",")
        for county in counties:
            name = county.strip()
            if name:
                unique_counties.add(name)

        started = parse_date(incident.get("Started", ""))
        if started and (latest_date is None or started > latest_date):
            latest_date = started

    return {
        "totalIncidents": len(incidents),
        "totalAcresBurned": round(total_acres, 2),
        "totalStructuresDestroyed": int(total_structures),
        "activeIncidents": active_incidents,
        "countiesImpacted": len(unique_counties),
        "latestIncidentDate": latest_date.strftime("%Y-%m-%d") if latest_date else "Unknown",
    }


class CalFireHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            self._send_json(self.server.summary_data)  # type: ignore[attr-defined]
            return
        if parsed.path == "/api/incidents":
            query = parse_qs(parsed.query)
            limit = 15
            if "limit" in query:
                try:
                    limit = max(1, min(int(query["limit"][0]), 100))
                except ValueError:
                    limit = 15
            self._send_json(self.server.sample_incidents[:limit])  # type: ignore[attr-defined]
            return
        super().do_GET()

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int) -> None:
    incidents = load_incidents(DATASET_PATH)
    summary = compute_summary(incidents)

    handler = partial(CalFireHandler, directory=str(WEB_DIR))
    with ThreadingHTTPServer((host, port), handler) as httpd:
        httpd.summary_data = summary  # type: ignore[attr-defined]
        httpd.sample_incidents = [
            {
                "Name": item.get("Name", ""),
                "Started": item.get("Started", ""),
                "County": item.get("Counties", ""),
                "AcresBurned": item.get("AcresBurned", ""),
                "Active": item.get("Active", ""),
            }
            for item in incidents
        ]  # type: ignore[attr-defined]
        print(f"CALFIRE website running at http://{host}:{port}")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CALFIRE local website.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the web server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the web server.")
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
