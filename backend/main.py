from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ml_engine import load_model, encode_applicant, explain_prediction, train_model

DB = Path(__file__).resolve().parent / "applications.db"
app = FastAPI(title="Bharat Capital Explainable Loan API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class LoanApplication(BaseModel):
    applicant_name: str = Field(min_length=2, max_length=100)
    city: str
    employment_type: str
    cibil_score: int = Field(ge=300, le=900)
    annual_income_inr: float = Field(gt=0)
    loan_amount_inr: float = Field(gt=0)
    dti_percentage: float = Field(ge=0, le=100)
    recent_defaults: int = Field(default=0, ge=0, le=20)

def connect():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS LoanApplications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT NOT NULL, city TEXT NOT NULL, employment_type TEXT NOT NULL,
            cibil_score INTEGER NOT NULL, annual_income_inr REAL NOT NULL, loan_amount_inr REAL NOT NULL,
            dti_percentage REAL NOT NULL, recent_defaults INTEGER NOT NULL,
            submitted_at TEXT NOT NULL, decision TEXT NOT NULL, denial_reason TEXT
        )""")
init_db()
bundle = load_model()

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "DecisionTreeClassifier"}

@app.post("/api/apply")
def apply_loan(application: LoanApplication):
    data = application.model_dump()
    encoded = encode_applicant(data, bundle["columns"])
    approved = int(bundle["model"].predict(encoded)[0]) == 1
    decision = "Approve" if approved else "Deny"
    reason = explain_prediction(bundle["model"], encoded, bundle["columns"], approved)
    timestamp = datetime.now(timezone.utc).isoformat()
    with connect() as con:
        cur = con.execute("""INSERT INTO LoanApplications
        (applicant_name, city, employment_type, cibil_score, annual_income_inr, loan_amount_inr,
        dti_percentage, recent_defaults, submitted_at, decision, denial_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (data["applicant_name"], data["city"], data["employment_type"], data["cibil_score"],
         data["annual_income_inr"], data["loan_amount_inr"], data["dti_percentage"],
         data["recent_defaults"], timestamp, decision, None if approved else reason))
        app_id = cur.lastrowid
    return {"id": app_id, "decision": decision, "denial_reason": None if approved else reason, "submitted_at": timestamp}

@app.get("/api/applications")
def applications():
    with connect() as con:
        rows = con.execute("SELECT * FROM LoanApplications ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

@app.post("/api/admin/login")
def admin_login(payload: dict):
    # Demo-only mock login. Replace with real authentication before deployment.
    if payload.get("username") == "admin" and payload.get("password") == "bharat123":
        return {"ok": True, "token": "demo-admin-session"}
    raise HTTPException(status_code=401, detail="Invalid demo credentials")

@app.post("/api/admin/retrain")
def retrain():
    accuracy = train_model()
    global bundle
    bundle = load_model()
    return {"ok": True, "test_accuracy": round(accuracy, 4)}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bharat Capital</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #eef2f8;
                color: #172554;
                margin: 0;
            }
            header {
                background: #12315b;
                color: white;
                text-align: center;
                padding: 25px;
            }
            main {
                max-width: 600px;
                margin: 25px auto;
                background: white;
                padding: 25px;
                border-radius: 12px;
            }
            label {
                display: block;
                margin-top: 14px;
                font-weight: bold;
            }
            input, select, button {
                box-sizing: border-box;
                width: 100%;
                padding: 11px;
                margin-top: 6px;
                border: 1px solid #ccd4e0;
                border-radius: 6px;
            }
            button {
                background: #174ea6;
                color: white;
                cursor: pointer;
                margin-top: 20px;
            }
            #result {
                margin-top: 20px;
                overflow-wrap: anywhere;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Bharat Capital</h1>
            <p>Explainable AI Loan Origination System</p>
        </header>

        <main>
            <h2>Apply for a Loan</h2>

            <form id="loanForm">
                <label>Applicant Name</label>
                <input name="applicant_name" required minlength="2">

                <label>City</label>
                <input name="city" required placeholder="e.g. Mumbai">

                <label>Employment Type</label>
                <select name="employment_type" required>
                    <option value="Salaried">Salaried</option>
                    <option value="Self-employed">Self-employed</option>
                </select>

                <label>CIBIL Score (300–900)</label>
                <input name="cibil_score" type="number" min="300" max="900" required>

                <label>Annual Income (₹)</label>
                <input name="annual_income_inr" type="number" min="1" required>

                <label>Loan Amount Requested (₹)</label>
                <input name="loan_amount_inr" type="number" min="1" required>

                <label>Debt-to-Income Ratio (%)</label>
                <input name="dti_percentage" type="number" min="0" max="100" step="0.1" required>

                <label>Recent Defaults</label>
                <input name="recent_defaults" type="number" min="0" max="20" value="0" required>

                <button type="submit">Evaluate Loan Application</button>
            </form>

            <div id="result"></div>
            <p><a href="/admin">Open Admin Dashboard →</a></p>
        </main>

        <script>
            document.getElementById('loanForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const form = Object.fromEntries(new FormData(this));
                for (const key of [
                    'cibil_score', 'annual_income_inr',
                    'loan_amount_inr', 'dti_percentage', 'recent_defaults'
                ]) {
                    form[key] = Number(form[key]);
                }

                const result = document.getElementById('result');
                result.textContent = 'Evaluating application...';

                try {
                    const response = await fetch('/api/apply', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(form)
                    });

                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(JSON.stringify(data.detail || data));
                    }

                    result.textContent =
                        'Decision: ' + data.decision +
                        ' | Application ID: ' + data.id +
                        ' | Reason: ' + (data.denial_reason ||
                        'The AI model approved this application.');

                    result.style.background =
                        data.decision === 'Approve' ? '#dcfce7' : '#fee2e2';
                    result.style.padding = '15px';
                } catch (error) {
                    result.textContent = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    """




@app.get("/admin", response_class=HTMLResponse)
def admin():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bharat Capital Admin Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #eef2f8;
                color: #172554;
                padding: 20px;
            }
            main {
                max-width: 1100px;
                margin: auto;
                background: white;
                padding: 24px;
                border-radius: 12px;
            }
            input, button {
                padding: 11px;
                margin: 5px;
                border: 1px solid #ccd4e0;
                border-radius: 6px;
            }
            button {
                background: #174ea6;
                color: white;
                cursor: pointer;
            }
            .cards {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
            }
            .card {
                background: #e5edff;
                padding: 18px;
                border-radius: 8px;
            }
            .scroll {
                overflow-x: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 9px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <main>
            <h1>Bharat Capital — Admin Dashboard</h1>
          

            <div id="login">
                <input id="username" placeholder="Username">
                <input id="password" type="password" placeholder="Password">
                <button onclick="login()">Login</button>
                <p id="message"></p>
            </div>

            <section id="dashboard" hidden>
                <div class="cards" id="stats"></div>
                <h2>Loan Applications</h2>

                <div class="scroll">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>City</th>
                                <th>CIBIL</th>
                                <th>Income</th>
                                <th>Loan</th>
                                <th>Decision</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody id="rows"></tbody>
                    </table>
                </div>
            </section>

            <p><a href="/">Back to Loan Application</a></p>
        </main>

        <script>
            async function login() {
                const message = document.getElementById('message');
                message.textContent = 'Checking login...';

                try {
                    const response = await fetch('/api/admin/login', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            username: document.getElementById('username').value,
                            password: document.getElementById('password').value
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Incorrect username or password');
                    }

                    const r = await fetch('/api/applications');
                    if (!r.ok) throw new Error('Could not load applications');

                    const data = await r.json();
                    const approved = data.filter(x => x.decision === 'Approve').length;
                    const rejected = data.filter(x => x.decision === 'Deny').length;

                    document.getElementById('stats').replaceChildren();

                    [
                        ['Total Applications', data.length],
                        ['Approved', approved],
                        ['Rejected', rejected]
                    ].forEach(([label, value]) => {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.textContent = label + ': ' + value;
                        document.getElementById('stats').appendChild(card);
                    });

                    const rows = document.getElementById('rows');
                    rows.replaceChildren();

                    data.forEach(x => {
                        const tr = document.createElement('tr');
                        [
                            x.id,
                            x.applicant_name,
                            x.city,
                            x.cibil_score,
                            '₹' + Number(x.annual_income_inr).toLocaleString('en-IN'),
                            '₹' + Number(x.loan_amount_inr).toLocaleString('en-IN'),
                            x.decision,
                            x.denial_reason || '—'
                        ].forEach(value => {
                            const td = document.createElement('td');
                            td.textContent = value ?? '';
                            tr.appendChild(td);
                        });
                        rows.appendChild(tr);
                    });

                    document.getElementById('login').hidden = true;
                    document.getElementById('dashboard').hidden = false;

                } catch (error) {
                    message.textContent = error.message;
                }
            }
        </script>
    </body>
    </html>
    """
