"""
Generate a PDF research report using ReportLab.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib import colors
import io
import os


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='SectionHeading', parent=styles['Heading1'],
        textColor=colors.HexColor('#2C3E50'), spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='SubHeading', parent=styles['Heading2'],
        textColor=colors.HexColor('#34495E'), spaceAfter=8
    ))
    return styles


def _df_to_table(df):
    data = [list(df.columns)] + df.values.tolist()
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.whitesmoke, colors.lightgrey]),
    ]))
    return t


def generate_pdf_report(output_path, dataset_summary, comparison_df,
                        threshold_rows, shap_importance_df, research_findings,
                        figures_dir=None):
    styles = _styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph(
        "Explainable and Reliable AI for Software Defect Prediction",
        styles['Title']
    ))
    story.append(Paragraph(
        "Comparative Study of Machine Learning Models", styles['Heading2']
    ))
    story.append(Spacer(1, 0.3 * inch))

    # Dataset Analysis
    story.append(Paragraph("1. Dataset Analysis", styles['SectionHeading']))
    ds_table = [['Metric', 'Value'],
                ['Total Records', str(dataset_summary['total_records'])],
                ['Total Features', str(dataset_summary['total_features'])],
                ['Defective Modules', str(dataset_summary['defective_count'])],
                ['Non-Defective Modules', str(dataset_summary['non_defective_count'])],
                ['Duplicates', str(dataset_summary['duplicates'])]]
    t = Table(ds_table, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Model Comparison
    story.append(Paragraph("2. Model Comparison", styles['SectionHeading']))
    story.append(_df_to_table(comparison_df))
    story.append(Spacer(1, 0.2 * inch))

    # Threshold Sensitivity
    story.append(Paragraph("3. Threshold Sensitivity", styles['SectionHeading']))
    import pandas as pd
    story.append(_df_to_table(pd.DataFrame(threshold_rows)))
    story.append(PageBreak())

    # SHAP
    story.append(Paragraph("4. SHAP Feature Importance", styles['SectionHeading']))
    story.append(_df_to_table(shap_importance_df.head(15)))
    story.append(Spacer(1, 0.2 * inch))

    # Findings
    story.append(Paragraph("5. Research Findings", styles['SectionHeading']))
    for q, ans in research_findings.items():
        story.append(Paragraph(f"<b>{q}</b>", styles['SubHeading']))
        story.append(Paragraph(ans, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return output_path