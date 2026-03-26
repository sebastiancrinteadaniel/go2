from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, HRFlowable
from svglib.svglib import svg2rlg
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

C_HEADER   = HexColor("#0f1923")
C_SECTION  = HexColor("#1e2d3d")
C_LIGHT_BG = HexColor("#f8fafc")
C_BORDER   = HexColor("#cbd5e1")
C_TEXT     = HexColor("#0f172a")
C_TEXT_MID = HexColor("#475569")
C_HDR_ROW  = HexColor("#e2e8f0")

PAGE_W, _ = A4
MARGIN = 15 * mm
CW = PAGE_W - 2 * MARGIN

_base = getSampleStyleSheet()

S_TITLE   = ParagraphStyle("RTitle",   parent=_base["Normal"], fontSize=15, fontName="Helvetica-Bold", textColor=white, spaceAfter=1*mm)
S_SUB     = ParagraphStyle("RSub",     parent=_base["Normal"], fontSize=7.5, fontName="Helvetica", textColor=HexColor("#94a3b8"))
S_SEC     = ParagraphStyle("RSec",     parent=_base["Normal"], fontSize=8.5, fontName="Helvetica-Bold", textColor=white)
S_CELL    = ParagraphStyle("RCell",    parent=_base["Normal"], fontSize=8, fontName="Helvetica", textColor=C_TEXT)
S_CELL_B  = ParagraphStyle("RCellB",   parent=_base["Normal"], fontSize=8, fontName="Helvetica-Bold", textColor=C_TEXT)
S_BODY    = ParagraphStyle("RBody",    parent=_base["Normal"], fontSize=8, fontName="Helvetica", textColor=C_TEXT, leftIndent=4*mm, spaceAfter=1.5*mm)
S_FOOTER  = ParagraphStyle("RFooter",  parent=_base["Normal"], fontSize=7, fontName="Helvetica", textColor=C_TEXT_MID, alignment=TA_CENTER)
S_SUMMARY = ParagraphStyle("RSummary", parent=_base["Normal"], fontSize=7.5, fontName="Helvetica-Oblique", textColor=C_TEXT_MID, alignment=TA_RIGHT)

JOINT_NAMES = ["FR_0","FR_1","FR_2","FL_0","FL_1","FL_2","RR_0","RR_1","RR_2","RL_0","RL_1","RL_2"]

_STATUS_COLORS = {
    "PRESENT": "#22c55e", "OK": "#22c55e", "CONNECTED": "#22c55e",
    "DAMAGED": "#f97316", "WARNING": "#f97316",
    "MISSING": "#ef4444", "CRITICAL": "#ef4444", "DISCONNECTED": "#ef4444", "LOW": "#ef4444", "HIGH": "#ef4444",
}

_counter_lock = threading.Lock()


@dataclass
class ReportData:
    report_id: str
    generated_at: datetime
    operator: str
    location: str
    mode: str
    cpu_percent: float
    ram_percent: float
    uptime_seconds: int
    battery_soc: Optional[int]
    robot_connected: bool
    frame_rate: float
    camera_source: str
    imu_roll_rad: Optional[float]
    imu_pitch_rad: Optional[float]
    imu_yaw_rad: Optional[float]
    detections: list
    motor_temps: list
    avg_temp_c: Optional[float]
    peak_temp_c: Optional[float]
    peak_joint_name: Optional[str]
    frame_jpeg: Optional[bytes]


def next_report_id() -> str:
    path = Path("reports/counter.txt")
    path.parent.mkdir(exist_ok=True)
    with _counter_lock:
        n = int(path.read_text().strip()) + 1 if path.exists() else 1
        path.write_text(str(n))
    return f"{datetime.now().year}-{n:03d}"


def _fmt(v, spec=".1f", fallback="--") -> str:
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return fallback


def _fmt_uptime(s: int) -> str:
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _deg(rad: Optional[float]) -> Optional[float]:
    return math.degrees(rad) if rad is not None else None


def _badge(text: str) -> Paragraph:
    color = _STATUS_COLORS.get(text, "#3b82f6")
    return Paragraph(f'<font color="{color}"><b>{text}</b></font>', S_CELL)


def _p(text, style=None) -> Paragraph:
    return Paragraph(str(text), style or S_CELL)


def _detection_status(det: dict) -> str:
    label = (det.get("class") or det.get("label") or "").lower()
    if any(k in label for k in ("missing", "absent")):
        return "MISSING"
    if any(k in label for k in ("damaged", "broken", "crack")):
        return "DAMAGED"
    return "PRESENT"


def _section_bar(title: str) -> list:
    tbl = Table([[_p(title, S_SEC)]], colWidths=[CW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_SECTION),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return [tbl, Spacer(1, 1 * mm)]


def _table(rows, col_widths, header=None) -> Table:
    data = []
    if header:
        data.append([_p(h, S_CELL_B) for h in header])
    for row in rows:
        data.append([c if isinstance(c, Paragraph) else _p(str(c)) for c in row])

    cmds = [
        ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), C_HDR_ROW),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    offset = 1 if header else 0
    for i in range(offset, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), C_LIGHT_BG))

    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(cmds))
    return t


def _enrich(detections: list) -> list:
    return [{**d, "_status": _detection_status(d)} for d in detections]


def _findings(data: ReportData, enriched: list) -> list:
    items = []
    for d in enriched:
        label = (d.get("class") or d.get("label") or "Unknown").replace("_", " ").title()
        if d["_status"] == "MISSING":
            items.append(f"<b>{label}</b> — not detected in inspection frame. Manual verification required.")
        elif d["_status"] == "DAMAGED":
            conf = d.get("conf") or d.get("confidence")
            conf_str = f", conf. {int(float(conf) * 100)}%" if conf is not None else ""
            items.append(f"<b>{label}</b> — detected as DAMAGED{conf_str}. Schedule maintenance before next cycle.")

    for i, temp in enumerate(data.motor_temps or []):
        j = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}"
        if temp > 55:
            items.append(f"<b>{j} CRITICAL</b> — {temp:.1f} °C exceeds critical threshold (55 °C). Immediate shutdown recommended.")
        elif temp > 40:
            items.append(f"<b>{j} thermal spike</b> — {temp:.1f} °C above warning threshold (40 °C). Monitor closely.")

    if data.battery_soc is not None and data.battery_soc < 20:
        items.append(f"<b>Low battery</b> — {data.battery_soc}% remaining. Return to charging station.")
    if not data.robot_connected:
        items.append("<b>Robot telemetry disconnected</b> — readings may be stale or unavailable.")
    return items


def _recommendations(data: ReportData, enriched: list) -> list:
    recs = []
    for d in enriched:
        label = (d.get("class") or d.get("label") or "Unknown").replace("_", " ").title()
        if d["_status"] == "MISSING":
            recs.append(f"Locate or replace {label} — not detected during inspection.")
        elif d["_status"] == "DAMAGED":
            recs.append(f"Inspect and service {label} — flagged as DAMAGED.")

    hot = [JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}" for i, t in enumerate(data.motor_temps or []) if t > 40]
    if hot:
        recs.append(f"Re-scan thermal zone around {', '.join(hot)} for heat source / insulation issue.")
    if data.battery_soc is not None and data.battery_soc < 20:
        recs.append("Return robot to charging station immediately.")
    if any(d["_status"] != "PRESENT" for d in enriched):
        recs.append("Re-run inspection after corrective actions are completed.")
    if not recs:
        recs.append("No corrective actions required. Schedule routine follow-up inspection.")
    return recs


def build_pdf(data: ReportData) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"QC Report QC-{data.report_id}", author=data.operator,
    )
    ts = data.generated_at.strftime("%Y-%m-%d %H:%M %Z")
    story = []

    logo_path = Path(__file__).parent.parent / "static/icons/logo.svg"
    logo_drawing = svg2rlg(str(logo_path)) if logo_path.exists() else None
    if logo_drawing:
        scale = 28 / logo_drawing.height
        logo_drawing.width  *= scale
        logo_drawing.height *= scale
        logo_drawing.transform = (scale, 0, 0, scale, 0, 0)
        logo_col_w = logo_drawing.width + 20
        logo_cell  = logo_drawing
    else:
        logo_col_w = 0
        logo_cell  = _p("")

    text_col_w = CW - logo_col_w
    header_tbl = Table([[
        logo_cell,
        [_p("QUADRUPED C2 — INDUSTRIAL INSPECTION QC REPORT", S_TITLE),
         _p(f"Report ID: QC-{data.report_id}  |  {ts}  |  Operator: {data.operator}", S_SUB)],
    ]], colWidths=[logo_col_w, text_col_w])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C_HEADER),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [header_tbl, Spacer(1, 4 * mm)]

    if data.frame_jpeg:
        img = Image(io.BytesIO(data.frame_jpeg))
        img.drawWidth, img.drawHeight = CW, CW * 9 / 16
        story += [img, Spacer(1, 4 * mm)]

    mode_label = {"hd_view": "HD VIEW", "go2": "GO2 CAMERA", "thermal": "THERMAL"}.get(data.mode, data.mode.upper())
    story += _section_bar("1. SESSION OVERVIEW")
    story += [_table([
        ["Report ID",       f"QC-{data.report_id}"],
        ["Date",            data.generated_at.strftime("%Y-%m-%d")],
        ["Time",            data.generated_at.strftime("%H:%M:%S %Z")],
        ["Operator",        data.operator],
        ["Location / Zone", data.location],
        ["Robot Unit",      "Unitree Go2"],
        ["Inspection Mode", mode_label],
    ], [CW * 0.35, CW * 0.65]), Spacer(1, 4 * mm)]

    conn_s = "CONNECTED" if data.robot_connected else "DISCONNECTED"
    batt_s = "LOW" if (data.battery_soc is not None and data.battery_soc < 20) else "OK"
    cpu_s  = "HIGH" if data.cpu_percent > 90 else "OK"
    pt     = data.peak_temp_c
    temp_s = "CRITICAL" if pt and pt > 55 else ("WARNING" if pt and pt > 40 else "OK")
    story += _section_bar("2. SYSTEM STATUS AT TIME OF CAPTURE")
    story += [_table([
        ["Connection",      conn_s,                                                           _badge(conn_s)],
        ["Battery",         f"{data.battery_soc}%" if data.battery_soc is not None else "--", _badge(batt_s)],
        ["CPU Usage",       f"{_fmt(data.cpu_percent)}%",                                     _badge(cpu_s)],
        ["RAM Usage",       f"{_fmt(data.ram_percent)}%",                                     _p("—")],
        ["Session Uptime",  _fmt_uptime(data.uptime_seconds),                                 _p("—")],
        ["Avg Motor Temp",  f"{_fmt(data.avg_temp_c)} °C",                                    _p("—")],
        ["Peak Motor Temp", f"{_fmt(data.peak_temp_c)} °C ({data.peak_joint_name or '--'})",  _badge(temp_s)],
    ], [CW * 0.40, CW * 0.35, CW * 0.25], header=["Parameter", "Value", "Status"]), Spacer(1, 4 * mm)]

    story += _section_bar("3. CAMERA / STREAM INFO")
    story += [_table([
        ["Camera Source", data.camera_source],
        ["Frame Rate",    f"{_fmt(data.frame_rate)} FPS"],
    ], [CW * 0.35, CW * 0.65]), Spacer(1, 4 * mm)]

    def _imu_row(axis, rad):
        d = _deg(rad)
        return [axis, f"{_fmt(d)}°" if d is not None else "--", f"{_fmt(rad, '.4f')} rad" if rad is not None else "--"]

    story += _section_bar("4. IMU ORIENTATION")
    story += [_table([
        _imu_row("Pitch", data.imu_pitch_rad),
        _imu_row("Roll",  data.imu_roll_rad),
        _imu_row("Yaw",   data.imu_yaw_rad),
    ], [CW * 0.25, CW * 0.375, CW * 0.375], header=["Axis", "Value (°)", "Value (rad)"]), Spacer(1, 4 * mm)]

    enriched = _enrich(data.detections)
    if enriched:
        det_rows = []
        for d in enriched:
            label = (d.get("class") or d.get("label") or "Unknown").replace("_", " ").title()
            conf  = d.get("conf") or d.get("confidence")
            det_rows.append([label, _badge(d["_status"]), f"{int(float(conf) * 100)}%" if conf is not None else "--", "—"])
        n_ok  = sum(1 for d in enriched if d["_status"] == "PRESENT")
        n_mis = sum(1 for d in enriched if d["_status"] == "MISSING")
        n_dmg = sum(1 for d in enriched if d["_status"] == "DAMAGED")
    else:
        det_rows = [["No detections in current session", _p("—"), "—", "—"]]
        n_ok = n_mis = n_dmg = 0

    story += _section_bar("5. INDUSTRIAL COMPONENT DETECTION")
    story.append(_table(det_rows, [CW * 0.40, CW * 0.20, CW * 0.20, CW * 0.20], header=["Component", "Status", "Confidence", "Notes"]))
    if enriched:
        story += [Spacer(1, 1 * mm), _p(f"Present: {n_ok}  |  Missing: {n_mis}  |  Damaged: {n_dmg}  |  Total: {len(enriched)}", S_SUMMARY)]
    story.append(Spacer(1, 4 * mm))

    if data.motor_temps:
        temp_rows = []
        for i, t in enumerate(data.motor_temps):
            j  = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}"
            ts_= "CRITICAL" if t > 55 else ("WARNING" if t > 40 else "OK")
            temp_rows.append([j, f"{_fmt(t)} °C", "< 40 °C", _badge(ts_)])
        avg_s = "WARNING" if (data.avg_temp_c and data.avg_temp_c > 35) else "OK"
        temp_rows += [
            ["— Average —",                        f"{_fmt(data.avg_temp_c)} °C",  "< 35 °C", _badge(avg_s)],
            [f"— Peak ({data.peak_joint_name or '--'}) —", f"{_fmt(data.peak_temp_c)} °C", "< 40 °C", _badge(temp_s)],
        ]
    else:
        temp_rows = [["No motor data available", "—", "—", "—"]]

    story += _section_bar("6. THERMAL ANALYSIS")
    story += [_table(temp_rows, [CW * 0.25, CW * 0.25, CW * 0.25, CW * 0.25], header=["Joint / Sensor", "Temperature", "Threshold", "Status"]), Spacer(1, 4 * mm)]

    story += _section_bar("7. FINDINGS & ANOMALIES")
    findings_list = _findings(data, enriched)
    for i, f in enumerate(findings_list, 1):
        story.append(Paragraph(f"{i}. {f}", S_BODY))
    if not findings_list:
        story.append(Paragraph("No anomalies detected.", S_BODY))
    story.append(Spacer(1, 4 * mm))

    story += _section_bar("8. RECOMMENDATIONS")
    for rec in _recommendations(data, enriched):
        story.append(Paragraph(f"&#9744;  {rec}", S_BODY))
    story.append(Spacer(1, 4 * mm))

    story += _section_bar("9. SIGN-OFF")
    story += [_table([
        ["Operator", data.operator, "", data.generated_at.strftime("%Y-%m-%d")],
        ["Reviewer", "—", "", "—"],
        ["QA Lead",  "—", "", "—"],
    ], [CW * 0.20, CW * 0.30, CW * 0.30, CW * 0.20], header=["Role", "Name", "Signature", "Date"]), Spacer(1, 6 * mm)]

    story += [
        HRFlowable(width=CW, color=C_BORDER),
        Spacer(1, 2 * mm),
        _p(f"Generated by Quadruped C2 Industrial Inspection Dashboard  |  {ts}", S_FOOTER),
    ]

    doc.build(story)
    return buf.getvalue()
