from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(12)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # Add a bottom border representation by just making it bold and uppercase
    
def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
        p.add_run(text)
    else:
        p.add_run(text)

doc = Document()

# Margins
sections = doc.sections
for section in sections:
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)

# Header
name = doc.add_paragraph()
name_run = name.add_run("MADEEHA TABASSUM")
name_run.bold = True
name_run.font.size = Pt(16)
name.alignment = WD_ALIGN_PARAGRAPH.CENTER
name.paragraph_format.space_after = Pt(0)

subtitle = doc.add_paragraph()
sub_run = subtitle.add_run("Data Analyst | AI & Data Science Undergraduate | Python Developer")
sub_run.font.size = Pt(11)
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(0)

contact = doc.add_paragraph()
contact_run = contact.add_run("Ballari, Karnataka, India | +91 96866 75159 | madeeha0717@gmail.com | linkedin.com/in/madeeha-tabassum | github.com/madeeha0717-git")
contact_run.font.size = Pt(9)
contact.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Summary
add_heading(doc, "PROFESSIONAL SUMMARY")
add_bullet(doc, "AI & Data Science undergraduate (Expected 2027, CGPA 8.51) with hands-on experience in Python, SQL, and data visualization (Tableau, Power BI) across machine learning, causal inference, and automation.")
add_bullet(doc, "Completed a GenAI-powered data analytics job simulation with Tata iQ, covering exploratory data analysis (EDA) and predictive modeling for customer delinquency risk.")

# Technical Skills
add_heading(doc, "TECHNICAL SKILLS")
add_bullet(doc, " Python (NumPy, Pandas, Matplotlib, Seaborn), Java, C, Object-Oriented Programming", bold_prefix="Programming & OOP:")
add_bullet(doc, " SQL, PostgreSQL, SQLite, DBMS Fundamentals, Data Cleaning & Wrangling, Exploratory Data Analysis (EDA)", bold_prefix="Data & Databases:")
add_bullet(doc, " Machine Learning, Causal Inference, Uplift Modeling, NLP, Predictive Modelling, Scikit-Learn, GenAI-Assisted Analysis", bold_prefix="AI & Machine Learning:")
add_bullet(doc, " Tableau, Power BI, Advanced Excel (Pivot Tables, Dashboards), Data Analytics", bold_prefix="Data Visualization & Analytics:")
add_bullet(doc, " Git, GitHub, Streamlit, Jupyter Notebook, Software Testing", bold_prefix="Tools & Platforms:")

# Projects
add_heading(doc, "TECHNICAL PROJECTS")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Marketing Campaign Causal Lift & Uplift Modeling Platform (Python, Scikit-Learn, Streamlit, SQLite)").bold = True
add_bullet(doc, "Built an end-to-end machine learning pipeline using a T-Learner (Gradient Boosting) to estimate the true causal Individual Treatment Effect (ITE) of marketing campaigns.")
add_bullet(doc, "Developed an interactive Streamlit dashboard and ROI simulator, demonstrating that Uplift Targeting yielded significantly higher expected profit compared to standard purchase-probability models.")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("HR Analytics Dashboard (Power BI, PostgreSQL, Python)").bold = True
add_bullet(doc, "Built an interactive multi-page Power BI dashboard analyzing employee attrition, demographics, job satisfaction, and compensation, using data sourced from PostgreSQL and cleaned with Python (Pandas) and Excel.")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Blockchain-Secured Web Application Firewall (WAF)").bold = True
add_bullet(doc, "Achieved 100% immutability of security logs by architecting a decentralized firewall that uses blockchain to store and manage access-control rules.")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Smart Irrigation System (IoT & Automation)").bold = True
add_bullet(doc, "Cut manual water-monitoring effort by 80% by engineering an automated irrigation system using Arduino and soil-moisture sensors.")


# Major Project
add_heading(doc, "MAJOR PROJECT")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("AI-Driven Multi-Disease Risk Assessment System for Diabetes, Heart Disease, Stroke, and Asthma with Intelligent Hospital Recommendation").bold = True
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Research Project (4-member team) | Guide: Dr. Mehboob Mujawar | Dept. of AI & Data Science").italic = True
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Technologies: ").bold = True
p.add_run("XGBoost, Machine Learning, Healthcare Analytics, Clinical Decision Support, Predictive Analytics, Data Preprocessing, Feature Engineering")
add_bullet(doc, "Co-developing an AI-driven multi-disease risk assessment framework, applying disease-specific XGBoost models to predict diabetes, heart disease, stroke, and asthma risk within a unified system with a planned hospital recommendation module.")
add_bullet(doc, "Completed data collection, preprocessing (missing-value imputation, duplicate removal, normalization), and feature engineering/selection across four disease-specific datasets sourced from Kaggle and the UCI ML Repository.")
add_bullet(doc, "Next phase: training and evaluating disease-specific XGBoost classifiers (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and building the hospital recommendation module.")

# Virtual Experience
add_heading(doc, "VIRTUAL EXPERIENCE")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("GenAI Powered Data Analytics Job Simulation | Tata iQ (via Forage) | July 2026").bold = True
add_bullet(doc, "Conducted exploratory data analysis (EDA) to assess data quality and identify risk indicators for Tata iQ's Financial Services team, and built a no-code predictive model for customer delinquency risk using GenAI-assisted analysis.")
add_bullet(doc, "Proposed an AI-driven collections strategy balancing automation, ethical AI principles, and regulatory compliance requirements.")

# Experience
add_heading(doc, "EXPERIENCE")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Web Development Intern | InternPe | Feb 2025 – Mar 2025").bold = True
add_bullet(doc, "Completed all assigned project tasks 15% ahead of deadline by converting UI designs into functional website modules (HTML, CSS, JavaScript) under structured SDLC practices.")

# Education
add_heading(doc, "EDUCATION")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Bearys Institute of Technology, Mangalore").bold = True
p.add_run(" | B.E. in Artificial Intelligence & Data Science | Expected 2027 | Current CGPA: 8.51")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Abdul Salam Memorial College, Ballari").bold = True
p.add_run(" | 12th Std (Science) | 2023 | Percentage: 82%")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Jawahar English Medium High School").bold = True
p.add_run(" | 10th Std | 2021 | Percentage: 77%")


# Certifications & Achievements
add_heading(doc, "CERTIFICATIONS & ACHIEVEMENTS")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Certifications: ").bold = True
p.add_run("NPTEL – Data Analytics with Python (Elite Silver Certification), 2024 · NPTEL – Software Testing, 2024")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Workshops & Achievements:").bold = True
add_bullet(doc, "3rd Place - Smart India Hackathon (SIH) Internal Hackathon")
add_bullet(doc, "Hackathon Participant – \"Make for Mangalore\" (Urban Civic-Tech Track)")
add_bullet(doc, "GDG Workshop – Data Structures, Algorithms & GenAI Technologies, 2024")

doc.save('Madeeha_Tabassum_Resume.docx')
print("Document created successfully.")
