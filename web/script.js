const summaryCards = document.getElementById("summary-cards");
const incidentsBody = document.getElementById("incidents-body");

function addCard(label, value) {
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `<h3>${label}</h3><p>${value}</p>`;
  summaryCards.appendChild(card);
}

async function loadSummary() {
  const response = await fetch("/api/summary");
  const data = await response.json();
  addCard("Total Incidents", data.totalIncidents);
  addCard("Active Incidents", data.activeIncidents);
  addCard("Acres Burned", data.totalAcresBurned.toLocaleString());
  addCard("Structures Destroyed", data.totalStructuresDestroyed);
  addCard("Counties Impacted", data.countiesImpacted);
  addCard("Latest Incident Date", data.latestIncidentDate);
}

async function loadIncidents() {
  const response = await fetch("/api/incidents?limit=12");
  const data = await response.json();
  data.forEach((incident) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${incident.Name || "-"}</td>
      <td>${incident.Started || "-"}</td>
      <td>${incident.County || "-"}</td>
      <td>${incident.AcresBurned || "0"}</td>
      <td>${incident.Active || "-"}</td>
    `;
    incidentsBody.appendChild(row);
  });
}

Promise.all([loadSummary(), loadIncidents()]).catch(() => {
  summaryCards.innerHTML = "<p>Unable to load incident data.</p>";
});
