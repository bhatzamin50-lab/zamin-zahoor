"""Smart Hotel Management & Analytics System - Flask app.
All data is stored in CSV files (data/ folder) and handled with Pandas."""
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
CHART_DIR = os.path.join(BASE, "static", "charts")

app = Flask(__name__)
app.secret_key = "student-project-key"  # needed for flash messages

# Replace "#" with the real portfolio URLs when they are supplied.
PORTFOLIO_LINKS = {"Zamin Zahoor": "https://zaminzahoor.netlify.app/", "Paul Zeeshan": "https://share.google/BvadBtb4EwfG3gSgX"}

COLUMNS = {
    "rooms": ["room_id", "room_number", "room_type", "price_per_night", "status"],
    "customers": ["customer_id", "customer_name", "phone", "email", "address"],
    "bookings": ["booking_id", "customer_id", "room_id", "check_in", "check_out",
                 "nights", "booking_status", "total_amount"],
    "billing": ["bill_id", "booking_id", "customer_name", "room_charges",
                "additional_charges", "total_bill", "payment_status", "payment_date"],
}
ROOM_TYPES = ["Single", "Double", "Deluxe", "Suite"]
ROOM_STATUS = ["Available", "Booked", "Maintenance"]
BOOKING_STATUS = ["Confirmed", "Checked-In", "Completed", "Cancelled"]
NAVY, GOLD = "#0b1f3a", "#c9a24b"


# ---------- CSV helpers ----------
def load(name):
    """Read a CSV into a DataFrame. Creates the file if missing; empty files give an empty table."""
    path = os.path.join(DATA_DIR, name + ".csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.DataFrame(columns=COLUMNS[name]).to_csv(path, index=False)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=COLUMNS[name])
    for col in COLUMNS[name]:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS[name]]


def save(name, df):
    """Write to a temporary file first, then replace the real file (safer saving)."""
    path = os.path.join(DATA_DIR, name + ".csv")
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def next_id(df, col, prefix):
    """Make a new unique ID such as R011 (highest number + 1)."""
    nums = pd.to_numeric(df[col].str.replace(prefix, "", regex=False), errors="coerce").dropna()
    n = int(nums.max()) + 1 if len(nums) else 1
    return f"{prefix}{n:03d}"


def num(df, col):
    """Column as numbers; bad or missing values become 0."""
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def parse_date(text):
    return datetime.strptime(text, "%Y-%m-%d").date()


def to_number(text):
    """Return a float or None if the text is not a valid number."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@app.template_filter("money")
def money(value):
    v = to_number(value)
    return "₹{:,.2f}".format(v if v is not None else 0)


def sync_rooms():
    """Rooms with a live booking today are 'Booked', others 'Available'. Maintenance stays as set."""
    rooms, bookings = load("rooms"), load("bookings")
    today = date.today().isoformat()
    live = bookings[bookings["booking_status"].isin(["Confirmed", "Checked-In"])
                    & (bookings["check_in"] <= today) & (bookings["check_out"] > today)]
    booked_ids = set(live["room_id"])
    new_status = rooms.apply(lambda r: r["status"] if r["status"] == "Maintenance"
                             else ("Booked" if r["room_id"] in booked_ids else "Available"), axis=1)
    if len(rooms) and not new_status.equals(rooms["status"]):
        rooms["status"] = new_status
        save("rooms", rooms)


def one(df, col, value):
    """First row of df where col == value, as a dict (or None)."""
    row = df[df[col] == value]
    return row.iloc[0].to_dict() if not row.empty else None


# ---------- Landing page and dashboard ----------
@app.route("/")
def index():
    return render_template("index.html", links=PORTFOLIO_LINKS)


def paid_revenue(bills):
    """Revenue = sum of Paid bills, calculated with a NumPy array."""
    amounts = np.array(num(bills, "total_bill"), dtype=float)
    paid = np.array(bills["payment_status"] == "Paid")
    return float(np.sum(amounts[paid])) if len(amounts) else 0.0


@app.route("/dashboard")
def dashboard():
    sync_rooms()
    rooms, bookings, customers, bills = load("rooms"), load("bookings"), load("customers"), load("billing")
    stats = {
        "Total Rooms": len(rooms),
        "Available Rooms": int((rooms["status"] == "Available").sum()),
        "Total Bookings": len(bookings),
        "Total Customers": len(customers),
        "Total Revenue": money(paid_revenue(bills)),
    }
    return render_template("dashboard.html", stats=stats)


# ---------- Rooms ----------
@app.route("/rooms")
def rooms():
    sync_rooms()
    df = load("rooms")
    status, rtype = request.args.get("status", ""), request.args.get("type", "")
    if status:
        df = df[df["status"] == status]
    if rtype:
        df = df[df["room_type"] == rtype]
    edit = one(load("rooms"), "room_id", request.args.get("edit", ""))
    return render_template("rooms.html", rooms=df.to_dict("records"), edit=edit, status=status,
                           rtype=rtype, types=ROOM_TYPES, statuses=ROOM_STATUS)


@app.route("/rooms/save", methods=["POST"])
def rooms_save():
    f, df = request.form, load("rooms")
    rid = f.get("room_id", "").strip()
    number, rtype, status = f.get("room_number", "").strip(), f.get("room_type"), f.get("status")
    price = to_number(f.get("price_per_night"))
    others = df[df["room_id"] != rid]
    if not number:
        flash("Room number is required.", "error")
    elif number in set(others["room_number"]):
        flash("That room number already exists.", "error")
    elif rtype not in ROOM_TYPES or status not in ROOM_STATUS:
        flash("Choose a valid room type and status.", "error")
    elif price is None or price <= 0:
        flash("Price per night must be a number greater than 0.", "error")
    else:
        row = {"room_id": rid or next_id(df, "room_id", "R"), "room_number": number,
               "room_type": rtype, "price_per_night": f"{price:g}", "status": status}
        if rid and rid in set(df["room_id"]):
            df.loc[df["room_id"] == rid, list(row)] = list(row.values())
            flash("Room updated.", "success")
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            flash("Room added.", "success")
        save("rooms", df)
    return redirect(url_for("rooms"))


@app.route("/rooms/<rid>/delete", methods=["POST"])
def rooms_delete(rid):
    if rid in set(load("bookings")["room_id"]):
        flash("This room has bookings, so it cannot be deleted. Set it to Maintenance instead.", "error")
    else:
        df = load("rooms")
        save("rooms", df[df["room_id"] != rid])
        flash("Room deleted.", "success")
    return redirect(url_for("rooms"))


# ---------- Customers ----------
@app.route("/customers")
def customers():
    df, q = load("customers"), request.args.get("q", "").strip().lower()
    if q:
        df = df[df["customer_name"].str.lower().str.contains(q, regex=False)
                | df["customer_id"].str.lower().str.contains(q, regex=False)]
    edit = one(load("customers"), "customer_id", request.args.get("edit", ""))
    return render_template("customers.html", customers=df.to_dict("records"), edit=edit, q=q)


@app.route("/customers/save", methods=["POST"])
def customers_save():
    f, df = request.form, load("customers")
    cid = f.get("customer_id", "").strip()
    name, phone = f.get("customer_name", "").strip(), f.get("phone", "").strip()
    email, address = f.get("email", "").strip(), f.get("address", "").strip()
    digits = "".join(c for c in phone if c.isdigit())
    if not name:
        flash("Customer name is required.", "error")
    elif not (7 <= len(digits) <= 15) or not all(c.isdigit() or c in "+- " for c in phone):
        flash("Enter a valid phone number (7 to 15 digits).", "error")
    elif "@" not in email or "." not in email.split("@")[-1]:
        flash("Enter a valid email address.", "error")
    else:
        row = {"customer_id": cid or next_id(df, "customer_id", "C"), "customer_name": name,
               "phone": phone, "email": email, "address": address}
        if cid and cid in set(df["customer_id"]):
            df.loc[df["customer_id"] == cid, list(row)] = list(row.values())
            flash("Customer updated.", "success")
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            flash("Customer added.", "success")
        save("customers", df)
    return redirect(url_for("customers"))


@app.route("/customers/<cid>/delete", methods=["POST"])
def customers_delete(cid):
    if cid in set(load("bookings")["customer_id"]):
        flash("This customer has bookings, so the record cannot be deleted.", "error")
    else:
        df = load("customers")
        save("customers", df[df["customer_id"] != cid])
        flash("Customer deleted.", "success")
    return redirect(url_for("customers"))


# ---------- Bookings ----------
@app.route("/bookings")
def bookings():
    df, status = load("bookings"), request.args.get("status", "")
    if status:
        df = df[df["booking_status"] == status]
    cust, rms = load("customers"), load("rooms")
    names = dict(zip(cust["customer_id"], cust["customer_name"]))
    numbers = dict(zip(rms["room_id"], rms["room_number"]))
    records = df.to_dict("records")
    for r in records:
        r["customer_name"] = names.get(r["customer_id"], "Unknown")
        r["room_number"] = numbers.get(r["room_id"], "?")
    edit = one(load("bookings"), "booking_id", request.args.get("edit", ""))
    return render_template("bookings.html", bookings=records, edit=edit, status=status,
                           customers=cust.to_dict("records"), rooms=rms.to_dict("records"),
                           statuses=BOOKING_STATUS, today=date.today().isoformat())


@app.route("/bookings/save", methods=["POST"])
def bookings_save():
    f, df = request.form, load("bookings")
    bid, cid, rid = f.get("booking_id", "").strip(), f.get("customer_id"), f.get("room_id")
    status = f.get("booking_status", "Confirmed")
    room = one(load("rooms"), "room_id", rid)
    back = url_for("bookings", edit=bid) if bid else url_for("bookings")
    try:
        ci, co = parse_date(f.get("check_in", "")), parse_date(f.get("check_out", ""))
    except ValueError:
        flash("Enter valid check-in and check-out dates.", "error")
        return redirect(back)
    if one(load("customers"), "customer_id", cid) is None or room is None:
        flash("Choose an existing customer and room.", "error")
    elif status not in BOOKING_STATUS:
        flash("Choose a valid booking status.", "error")
    elif co <= ci:
        flash("Check-out must be after check-in.", "error")
    elif not bid and ci < date.today():
        flash("Check-in date cannot be in the past.", "error")
    elif status != "Cancelled" and room["status"] == "Maintenance":
        flash("This room is under maintenance and cannot be booked.", "error")
    else:
        # Overlap rule: two stays clash if new_in < old_out and new_out > old_in
        same = df[(df["room_id"] == rid) & (df["booking_status"] != "Cancelled") & (df["booking_id"] != bid)]
        clash = same[(same["check_in"] < co.isoformat()) & (same["check_out"] > ci.isoformat())]
        if status != "Cancelled" and not clash.empty:
            flash(f"Room already reserved for those dates (booking {clash.iloc[0]['booking_id']}).", "error")
        else:
            nights = (co - ci).days
            total = float(np.multiply(float(room["price_per_night"]), nights))
            row = {"booking_id": bid or next_id(df, "booking_id", "B"), "customer_id": cid, "room_id": rid,
                   "check_in": ci.isoformat(), "check_out": co.isoformat(), "nights": str(nights),
                   "booking_status": status, "total_amount": f"{total:g}"}
            if bid and bid in set(df["booking_id"]):
                df.loc[df["booking_id"] == bid, list(row)] = list(row.values())
                fix_pending_bill(row)
                flash("Booking updated.", "success")
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                flash(f"Booking created. Total cost {money(total)} for {nights} night(s).", "success")
            save("bookings", df)
            sync_rooms()
    return redirect(url_for("bookings"))


def fix_pending_bill(row):
    """If a booking's cost changes, refresh its Pending bill so the numbers stay consistent."""
    bills = load("billing")
    mask = (bills["booking_id"] == row["booking_id"]) & (bills["payment_status"] == "Pending")
    if mask.any():
        bills.loc[mask, "room_charges"] = row["total_amount"]
        bills.loc[mask, "total_bill"] = (num(bills, "room_charges") + num(bills, "additional_charges"))[mask].map(lambda v: f"{v:g}")
        save("billing", bills)


@app.route("/bookings/<bid>/cancel", methods=["POST"])
def bookings_cancel(bid):
    df = load("bookings")
    df.loc[df["booking_id"] == bid, "booking_status"] = "Cancelled"
    save("bookings", df)
    sync_rooms()
    flash("Booking cancelled. The room is free for those dates again.", "success")
    return redirect(url_for("bookings"))


# ---------- Billing ----------
@app.route("/billing")
def billing():
    bills, bookings_df = load("billing"), load("bookings")
    billed = set(bills["booking_id"])
    open_bookings = bookings_df[(bookings_df["booking_status"] != "Cancelled") & ~bookings_df["booking_id"].isin(billed)]
    names = dict(zip(load("customers")["customer_id"], load("customers")["customer_name"]))
    options = [{**b, "customer_name": names.get(b["customer_id"], "Unknown")} for b in open_bookings.to_dict("records")]
    return render_template("billing.html", bills=bills.to_dict("records"), options=options)


@app.route("/billing/add", methods=["POST"])
def billing_add():
    bid, extra = request.form.get("booking_id"), to_number(request.form.get("additional_charges") or "0")
    bookings_df, bills = load("bookings"), load("billing")
    booking = one(bookings_df, "booking_id", bid)
    if booking is None or booking["booking_status"] == "Cancelled":
        flash("Choose a valid, non-cancelled booking.", "error")
    elif bid in set(bills["booking_id"]):
        flash("That booking already has a bill.", "error")
    elif extra is None or extra < 0:
        flash("Additional charges must be 0 or more.", "error")
    else:
        customer = one(load("customers"), "customer_id", booking["customer_id"])
        room_charges = float(booking["total_amount"])  # room charges come from the booking automatically
        row = {"bill_id": next_id(bills, "bill_id", "BL"), "booking_id": bid,
               "customer_name": customer["customer_name"] if customer else "Unknown",
               "room_charges": f"{room_charges:g}", "additional_charges": f"{extra:g}",
               "total_bill": f"{room_charges + extra:g}", "payment_status": "Pending", "payment_date": ""}
        save("billing", pd.concat([bills, pd.DataFrame([row])], ignore_index=True))
        flash(f"Bill created. Final bill {money(room_charges + extra)}.", "success")
    return redirect(url_for("billing"))


@app.route("/billing/<bill_id>/status", methods=["POST"])
def billing_status(bill_id):
    status, bills = request.form.get("status"), load("billing")
    if status in ("Paid", "Pending") and bill_id in set(bills["bill_id"]):
        bills.loc[bills["bill_id"] == bill_id, "payment_status"] = status
        bills.loc[bills["bill_id"] == bill_id, "payment_date"] = date.today().isoformat() if status == "Paid" else ""
        save("billing", bills)
        flash(f"Bill {bill_id} marked {status}.", "success")
    return redirect(url_for("billing"))


# ---------- Analytics ----------
def chart(name, fig):
    """Save a figure into static/charts and return its file name."""
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, name), dpi=110)
    plt.close(fig)
    return name


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color=NAVY, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def label_bars(ax):
    for c in ax.containers:
        ax.bar_label(c, padding=2)


@app.route("/analytics")
def analytics():
    sync_rooms()
    rooms_df, bookings_df, bills = load("rooms"), load("bookings"), load("billing")
    sns.set_theme(style="whitegrid")
    charts = {}

    # Numbers from NumPy arrays
    amounts = np.array(num(bills, "total_bill"), dtype=float)
    summary = {"Total revenue (paid bills)": money(paid_revenue(bills)),
               "Total bookings": len(bookings_df),
               "Average bill": money(amounts.mean() if len(amounts) else 0),
               "Highest bill": money(amounts.max() if len(amounts) else 0),
               "Pending amount": money(float(np.sum(amounts[np.array(bills["payment_status"] == "Pending")])) if len(amounts) else 0)}

    if len(rooms_df):  # available vs booked rooms
        counts = rooms_df["status"].value_counts().reindex(ROOM_STATUS, fill_value=0)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        sns.barplot(x=counts.index, y=counts.values, color=NAVY, ax=ax)
        style(ax, "Available vs booked rooms", "Room status", "Rooms")
        label_bars(ax)
        charts["rooms"] = chart("rooms_status.png", fig)

    paid = bills[(bills["payment_status"] == "Paid") & (bills["payment_date"].str.len() >= 7)]
    if len(paid):  # revenue by month (Pandas groupby)
        monthly = paid.assign(month=paid["payment_date"].str[:7], amount=num(paid, "total_bill")).groupby("month")["amount"].sum()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(monthly.index, monthly.values, color=GOLD)
        style(ax, "Revenue by month", "Month", "Revenue (Rs.)")
        label_bars(ax)
        charts["revenue"] = chart("revenue_month.png", fig)

    active = bookings_df[bookings_df["booking_status"] != "Cancelled"].merge(rooms_df, on="room_id", how="left")
    if len(active):  # bookings by room type
        by_type = active["room_type"].fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        sns.barplot(x=by_type.index, y=by_type.values, color=NAVY, ax=ax)
        style(ax, "Bookings by room type (not cancelled)", "Room type", "Bookings")
        label_bars(ax)
        charts["types"] = chart("bookings_type.png", fig)

    if len(bookings_df):  # booking status distribution
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        sns.countplot(data=bookings_df, x="booking_status", order=BOOKING_STATUS, color=GOLD, ax=ax)
        style(ax, "Booking status distribution", "Status", "Bookings")
        label_bars(ax)
        charts["booking_status"] = chart("booking_status.png", fig)

    if len(bills):  # payment status distribution
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        sns.countplot(data=bills, x="payment_status", order=["Paid", "Pending"], color=NAVY, ax=ax)
        style(ax, "Payment status distribution", "Payment status", "Bills")
        label_bars(ax)
        charts["payment"] = chart("payment_status.png", fig)

    return render_template("analytics.html", summary=summary, charts=charts, stamp=int(datetime.now().timestamp()))


if __name__ == "__main__":
    app.run(debug=True)
