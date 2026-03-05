from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import tempfile
from typing import List, Dict

def generate_report(memory: List[Dict], filename_context: str = "Session") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#0055ff'), spaceAfter=20)
    question_style = ParagraphStyle('QuestionStyle', parent=styles['Normal'], fontSize=12, textColor=colors.royalblue, spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold')
    answer_style = ParagraphStyle('AnswerStyle', parent=styles['Normal'], fontSize=11, textColor=colors.darkslategray, spaceAfter=15, leading=14)
    
    story = []
    story.append(Paragraph(f"Study Assistant Report: {filename_context}", title_style))
    story.append(Spacer(1, 10))
    
    for i in range(0, len(memory), 2):
        if i < len(memory) and memory[i]['role'] == 'user':
            q_text = memory[i]['text']
            story.append(Paragraph(f"Q: {q_text}", question_style))
            
        if i + 1 < len(memory) and memory[i+1]['role'] == 'assistant':
            a_text = memory[i+1]['text']
            story.append(Paragraph(a_text, answer_style))
            story.append(Spacer(1, 10))
            
    if not memory:
        story.append(Paragraph("No questions asked yet in this session.", answer_style))
        
    doc.build(story)
    return tmp.name
