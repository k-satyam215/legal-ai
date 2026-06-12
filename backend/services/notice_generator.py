"""Legal notice generator — text + PDF output."""
import os
import logging
from datetime import datetime
from backend.core.llm import call_llm
from backend.core.prompts import NOTICE_SYSTEM, NOTICE_USER

logger = logging.getLogger(__name__)

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_notice_text(
    notice_type, sender_name, sender_address,
    recipient_name, recipient_address, facts, relief, law,
) -> str:
    try:
        return call_llm(
            messages=[
                {"role": "system", "content": NOTICE_SYSTEM},
                {"role": "user", "content": NOTICE_USER.format(
                    notice_type=notice_type, sender_name=sender_name,
                    sender_address=sender_address, recipient_name=recipient_name,
                    recipient_address=recipient_address, facts=facts,
                    relief=relief, law=law,
                )},
            ]
        )
    except Exception as e:
        logger.error(f"[NoticeGenerator] {e}")
        date = datetime.now().strftime("%d %B %Y")
        return f"""LEGAL NOTICE\n\nDate: {date}\n\nFrom:\n{sender_name}\n{sender_address}\n\nTo:\n{recipient_name}\n{recipient_address}\n\nWHEREAS, {facts}\n\nAND WHEREAS the undersigned is entitled to {relief} under {law}.\n\nNOW THEREFORE, I call upon you to comply within 15 days from receipt of this notice, failing which legal proceedings shall be initiated.\n\nYours faithfully,\n{sender_name}\nDate: {date}"""


def generate_notice_pdf(notice_text: str, sender_name: str, output_path: str = "legal_notice.pdf") -> str:
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    doc    = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()

    title_style  = ParagraphStyle("T", parent=styles["Title"],  alignment=TA_CENTER, spaceAfter=20, fontSize=14)
    normal_style = ParagraphStyle("N", parent=styles["Normal"], spaceAfter=10, leading=18, alignment=TA_JUSTIFY)
    bold_style   = ParagraphStyle("B", parent=styles["Normal"], fontName="Helvetica-Bold", spaceAfter=6)

    content = [Paragraph("LEGAL NOTICE", title_style), Spacer(1, 10)]
    for line in notice_text.split("\n"):
        line = line.strip()
        if not line:
            content.append(Spacer(1, 6))
        elif line.isupper() and len(line) < 80:
            content.append(Paragraph(line, bold_style))
        else:
            content.append(Paragraph(line, normal_style))

    doc.build(content)
    logger.info(f"[NoticeGenerator] PDF saved: {output_path}")
    return output_path
