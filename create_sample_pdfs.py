"""
Convert text files to PDF for upload to Healthcare Q&A Assistant
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    print("✅ ReportLab available - creating PDF")
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("⚠️ ReportLab not available - will create simple PDF alternative")
    REPORTLAB_AVAILABLE = False

import os
from pathlib import Path

def create_pdf_with_reportlab(text_file_path, output_pdf_path):
    """Create PDF using ReportLab library."""
    # Read the text file
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        fontName='Helvetica-Bold',
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 12
    
    # Build story
    story = []
    
    # Split content into lines and process
    lines = content.split('\n')
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        
        if not line:  # Empty line
            if current_paragraph:
                story.append(Paragraph(current_paragraph, normal_style))
                story.append(Spacer(1, 6))
                current_paragraph = ""
            continue
        
        # Check if line is a header (all caps or ends with colon)
        if line.isupper() or line.endswith(':'):
            if current_paragraph:
                story.append(Paragraph(current_paragraph, normal_style))
                current_paragraph = ""
            
            # Create header
            header_style = ParagraphStyle(
                'CustomHeader',
                parent=styles['Heading2'],
                fontSize=12,
                fontName='Helvetica-Bold',
                spaceAfter=12,
                spaceBefore=12
            )
            story.append(Paragraph(line, header_style))
        else:
            # Add to current paragraph
            if current_paragraph:
                current_paragraph += " " + line
            else:
                current_paragraph = line
    
    # Add final paragraph if exists
    if current_paragraph:
        story.append(Paragraph(current_paragraph, normal_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF created: {output_pdf_path}")

def create_simple_pdf_alternative(text_file_path, output_pdf_path):
    """Create a simple HTML file that can be printed to PDF."""
    # Read the text file
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Medical Document</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 1in;
            font-size: 12pt;
        }}
        .header {{
            font-weight: bold;
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .title {{
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }}
        @page {{
            margin: 1in;
        }}
    </style>
</head>
<body>
"""
    
    # Process content
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            html_content += "<br><br>\n"
        elif line.isupper() or line.endswith(':'):
            html_content += f'<div class="header">{line}</div>\n'
        else:
            html_content += f'<p>{line}</p>\n'
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    html_file_path = output_pdf_path.replace('.pdf', '.html')
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f"✅ HTML file created: {html_file_path}")
    print("📝 To convert to PDF:")
    print("   1. Open the HTML file in your browser")
    print("   2. Press Ctrl+P (or Cmd+P on Mac)")
    print("   3. Select 'Save as PDF' as destination")
    print(f"   4. Save as: {output_pdf_path}")

def main():
    """Convert sample text files to PDF."""
    documents_dir = Path("data/documents")
    
    # List of text files to convert
    text_files = [
        "medication_administration_protocol.txt",
        "diabetes_management_guideline.txt", 
        "hypertension_protocol.txt"
    ]
    
    print("🏥 Converting Healthcare Documents to PDF")
    print("=" * 50)
    
    for text_file in text_files:
        text_path = documents_dir / text_file
        pdf_path = documents_dir / text_file.replace('.txt', '.pdf')
        
        if not text_path.exists():
            print(f"⚠️ File not found: {text_path}")
            continue
        
        print(f"\n📄 Processing: {text_file}")
        
        try:
            if REPORTLAB_AVAILABLE:
                create_pdf_with_reportlab(str(text_path), str(pdf_path))
            else:
                create_simple_pdf_alternative(str(text_path), str(pdf_path))
        except Exception as e:
            print(f"❌ Error processing {text_file}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Conversion complete!")
    print(f"📁 PDF files are in: {documents_dir}")
    print("\n🚀 Next steps:")
    print("1. Open the Healthcare Q&A Assistant")
    print("2. Go to the sidebar 'Document Management'")
    print("3. Upload the PDF files")
    print("4. Click 'Process Documents'")
    print("5. Initialize the system and start asking questions!")

if __name__ == "__main__":
    # Try to install reportlab if not available
    if not REPORTLAB_AVAILABLE:
        try:
            import subprocess
            import sys
            print("📦 Installing ReportLab...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
            print("✅ ReportLab installed successfully!")
            print("🔄 Please run this script again to create PDFs.")
        except Exception as e:
            print(f"⚠️ Could not install ReportLab: {e}")
            print("📄 Will create HTML files instead.")
            main()
    else:
        main()