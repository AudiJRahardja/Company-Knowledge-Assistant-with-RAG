"""Mock company database (SQLite) that the tools read and write.

    python -m rag.mockdb          (re)create storage/mock/brightnest.sqlite with the demo data

Tables
    employees   who people are, their role, manager and leave balances (internal tools)
    orders      customer orders with the e-mail address used to verify the customer (external tools)
    stores      opening hours per store (external tool; same data as the "Store Locations" document)
    tickets     HR / IT tickets created by the internal assistant
    handoffs    requests for a person to contact a customer, and sales leads (external assistant)

All people, orders and e-mail addresses are made up. The demo logins in data/mock/users.yaml map to
employees through the `username` column.
"""

import sqlite3
from datetime import date
from pathlib import Path

SCHEMA = """
CREATE TABLE employees (
    employee_id TEXT PRIMARY KEY,
    username TEXT UNIQUE,              -- demo login, or NULL for people without a demo account
    name TEXT NOT NULL,
    role TEXT NOT NULL,                -- employee | manager | hr
    title TEXT NOT NULL,
    email TEXT NOT NULL,
    location TEXT NOT NULL,
    manager_id TEXT REFERENCES employees(employee_id),
    join_date TEXT NOT NULL,
    annual_entitlement INTEGER NOT NULL,   -- this year's days, including the long-service bonus
    annual_taken REAL NOT NULL,
    carried_over INTEGER NOT NULL,         -- days carried over from last year
    carry_over_expires TEXT NOT NULL,      -- carried-over days must be used by this date (31 March)
    sick_days_taken INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    delivery_address TEXT NOT NULL,     -- never returned by the tool; only the city is
    delivery_city TEXT NOT NULL,
    placed_on TEXT NOT NULL,
    items TEXT NOT NULL,
    total_rp INTEGER NOT NULL,
    status TEXT NOT NULL,              -- processing | shipped | delivered | cancelled
    courier TEXT,
    tracking_number TEXT,
    estimated_delivery TEXT
);
CREATE TABLE stores (
    store_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    address TEXT NOT NULL,
    hours_json TEXT NOT NULL           -- {"mon": ["10:00", "22:00"], ... , "sun": null if closed}
);
CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY,
    team TEXT NOT NULL,                -- hr | it
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    summary TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE handoffs (
    reference TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                -- support | lead
    name TEXT NOT NULL,
    contact TEXT NOT NULL,
    company TEXT,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

EMPLOYEES = [
    # id, username, name, role, title, email, location, manager, joined, entitlement, taken, carried, expires, sick
    ("E001", None, "David Kusuma", "manager", "Chief Executive Officer and co-founder", "david.kusuma@brightnest.co.id",
     "Head office", None, "2012-01-01", 15, 4, 0, "2026-03-31", 0),
    ("E002", None, "Hendra Gunawan", "manager", "Chief Financial Officer", "hendra.gunawan@brightnest.co.id",
     "Head office", "E001", "2017-06-01", 14, 7, 0, "2026-03-31", 1),
    ("E003", "maya", "Maya Putri", "manager", "Finance Manager", "maya.putri@brightnest.co.id",
     "Head office", "E002", "2019-04-15", 14, 6.5, 3, "2026-03-31", 2),
    ("E004", "sari", "Sari Wulandari", "hr", "Head of People", "sari.wulandari@brightnest.co.id",
     "Head office", "E001", "2016-09-01", 15, 9, 5, "2026-03-31", 0),
    ("E005", None, "Bambang Susilo", "manager", "Area Manager, Jakarta stores", "bambang.susilo@brightnest.co.id",
     "Jakarta", "E001", "2018-03-01", 14, 10, 0, "2026-03-31", 3),
    ("E006", None, "Andi Wijaya", "manager", "Store Manager, Kemang", "andi.wijaya@brightnest.co.id",
     "Kemang store", "E005", "2018-05-01", 14, 8, 2, "2026-03-31", 1),
    ("E007", "rizky", "Rizky Pratama", "employee", "Sales Associate, Kemang", "rizky.pratama@brightnest.co.id",
     "Kemang store", "E006", "2023-02-01", 13, 5, 2, "2026-03-31", 2),
]

ORDERS = [
    ("BN-2026-104233", "rina.setiawati@example.com", "Rina Setiawati", "Jl. Cempaka No. 9", "Bandung",
     "2026-11-18", "Brightnest Kitchen 28 cm wok (1), Nest Living bath towel set (2)", 1_148_000, "shipped",
     "JNE", "JNE7783211045", "2026-11-25"),
    ("BN-2026-104501", "dewi.lestari@example.com", "Dewi Lestari", "Jl. Merpati No. 21", "Surabaya",
     "2026-11-21", "Air fryer 4.5 L (1)", 1_299_000, "processing", None, None, "2026-11-27"),
    ("BN-2026-103877", "agus.santoso@example.com", "Agus Santoso", "Jl. Kenanga No. 3", "Jakarta",
     "2026-11-05", "Dining table, oak, 6 seats (1)", 7_450_000, "delivered", "Brightnest delivery team", None,
     "2026-11-14"),
    ("BN-2026-104620", "putri.ananda@example.com", "Putri Ananda", "Jl. Flamboyan No. 12", "Medan",
     "2026-11-20", "Stand mixer (1)", 2_150_000, "cancelled", None, None, None),
]

_WEEK = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _hours(default, **overrides):
    return {day: overrides.get(day, default) for day in _WEEK}


STORES = [
    ("kemang", "Kemang", "Jakarta", "Jl. Kemang Raya No. 21, South Jakarta", _hours(["10:00", "22:00"])),
    ("kelapa-gading", "Kelapa Gading", "Jakarta", "Jl. Boulevard Raya Blok M3 No. 5, North Jakarta",
     _hours(["10:00", "22:00"])),
    ("bsd", "BSD", "Tangerang", "Jl. BSD Grand Boulevard No. 12, Tangerang", _hours(["10:00", "21:00"])),
    ("dago", "Dago", "Bandung", "Jl. Ir. H. Juanda No. 140, Bandung", _hours(["09:00", "21:00"], sun=["10:00", "20:00"])),
    ("pasteur", "Pasteur Outlet", "Bandung", "Jl. Dr. Djunjunan No. 77, Pasteur, Bandung",
     _hours(["09:00", "18:00"], mon=None)),
    ("darmo", "Darmo", "Surabaya", "Jl. Raya Darmo No. 45, Surabaya", _hours(["10:00", "21:00"])),
]

# Holiday rules from the "Store Locations and Opening Hours" document, for 2026.
IDUL_FITRI_DAY_1 = date(2026, 3, 20)
IDUL_FITRI_DAY_2 = date(2026, 3, 21)
EARLY_CLOSING_DAYS = {(12, 24), (12, 31)}  # all stores close at 18:00


def create(path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", EMPLOYEES)
        db.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", ORDERS)
        db.executemany("INSERT INTO stores VALUES (?,?,?,?,?)",
                       [(sid, name, city, addr, json.dumps(hours)) for sid, name, city, addr, hours in STORES])
    return path


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        create(path)
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


if __name__ == "__main__":
    from rag.config import get_settings

    created = create(get_settings().mock_db_path)
    print(f"Created {created} with {len(EMPLOYEES)} employees, {len(ORDERS)} orders and {len(STORES)} stores.")
