# CALFIRE

California wildfire incident analysis project with a lightweight local website dashboard.

## Repository contents

- `CALFIRE(RF) RN7945.ipynb`: Original ML notebook (Random Forest workflow)
- `California_Fire_Incidents.csv`: Source wildfire incidents dataset
- `app.py`: Local HTTP server exposing dashboard + JSON APIs
- `web/`: Static website files

## Run locally (website)

Prerequisite: Python 3.10+

```bash
cd /path/to/CALFIRE
python app.py --host 127.0.0.1 --port 8000
```

Then open: `http://127.0.0.1:8000`

Available API endpoints:
- `GET /api/summary` - aggregated incident metrics
- `GET /api/incidents?limit=12` - incident rows for table display

## Run notebook dependencies

```bash
pip install -r requirements.txt
```
