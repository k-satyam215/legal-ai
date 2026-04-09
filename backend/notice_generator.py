from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from datetime import datetime


def generate_notice(data, user_info, filename="legal_notice.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=20
    )

    normal_style = ParagraphStyle(
        name="NormalStyle",
        parent=styles["Normal"],
        spaceAfter=10,
        leading=16
    )

    bold_style = ParagraphStyle(
        name="BoldStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        spaceAfter=10
    )

    content = []

    # Title
    content.append(Paragraph("LEGAL NOTICE", title_style))

    # Date
    content.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y')}", normal_style))

    # From
    content.append(Paragraph("<b>From:</b>", bold_style))
    content.append(Paragraph(f"{user_info['name']}<br/>{user_info['address']}", normal_style))

    # To
    content.append(Paragraph("<b>To:</b>", bold_style))
    content.append(Paragraph(f"{user_info['landlord_name']}<br/>{user_info['landlord_address']}", normal_style))

    # Subject
    content.append(Paragraph("<b>Subject:</b> Legal Notice for Refund of Security Deposit", bold_style))
    content.append(Spacer(1, 15))

    # Opening
    content.append(Paragraph("Sir/Madam,", normal_style))

    # Replace placeholders
    for point in data.get("notice_points", []):
        text = point.replace("[amount]", user_info["amount"])
        content.append(Paragraph(text, normal_style))

    # Closing
    content.append(Spacer(1, 20))
    content.append(Paragraph("Yours sincerely,", normal_style))
    content.append(Paragraph(user_info["name"], normal_style))

    doc.build(content)

    print(f"✅ PDF generated: {filename}")