import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# 1. Load your saved Model and Scaler
voting_model = joblib.load('stroke_voting_model.pkl')
scaler = joblib.load('stroke_scaler.pkl')

# 2. Define the Columns
all_columns = [
    'age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 
    'gender_Male', 'ever_married_Yes', 'Residence_type_Urban', 
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
    'comorbidity_index', 'age_glucose_interaction'
]

engineered_cols = ['comorbidity_index', 'age_glucose_interaction']
user_cols = [col for col in all_columns if col not in engineered_cols]

# Groupings for the UI
demo_keywords = ['age', 'gender', 'residence', 'married']
vital_keywords = ['glucose', 'bmi', 'hypertension', 'heart']
work_keywords = ['work']
lifestyle_keywords = ['smoking']

group_demographics = [c for c in user_cols if any(k in c.lower() for k in demo_keywords)]
group_vitals = [c for c in user_cols if any(k in c.lower() for k in vital_keywords)]
group_work = [c for c in user_cols if any(k in c.lower() for k in work_keywords)]
group_lifestyle = [c for c in user_cols if any(k in c.lower() for k in lifestyle_keywords)]
# 🚨 RESTORED THIS MISSING LINE!
group_other = [c for c in user_cols if c not in group_demographics + group_vitals + group_work + group_lifestyle]

# 3. The Prediction Function
def predict_stroke(*args):
    cleaned_args = []
    for val in args:
        if isinstance(val, str) and val.startswith(('0', '1')):
            cleaned_args.append(int(val[0]))
        else:
            cleaned_args.append(val)
            
    data_dict = dict(zip(user_cols, cleaned_args))
    
    age = data_dict.get('age', 0)
    glucose = data_dict.get('avg_glucose_level', 0)
    heart = data_dict.get('heart_disease', 0)
    hyper = data_dict.get('hypertension', 0)
    
    data_dict['comorbidity_index'] = heart + hyper
    data_dict['age_glucose_interaction'] = age * glucose
    
    input_df = pd.DataFrame([data_dict], columns=all_columns)
    
    # Scaling
    continuous_cols = ['age', 'avg_glucose_level', 'bmi']
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    probability = voting_model.predict_proba(input_df)[0][1]
    
    if probability >= 0.50: 
        return f"🚨 HIGH RISK DETECTED\n\nProbability of Stroke: {probability*100:.1f}%\nRecommendation: Immediate medical consultation advised."
    elif probability >= 0.20: 
        return f"⚠️ MODERATE RISK\n\nProbability of Stroke: {probability*100:.1f}%\nRecommendation: Schedule a routine check-up and monitor vitals."
    else:
        return f"✅ LOW RISK DETECTED\n\nProbability of Stroke: {probability*100:.1f}%\nRecommendation: Maintain healthy lifestyle habits."

# 4. Helper Function
def create_friendly_input(col_name):
    if col_name == 'age':
        return gr.Number(label="Age (Years)", value=50)
    elif 'glucose' in col_name.lower():
        return gr.Number(label="Average Glucose Level", value=100.0)
    elif 'bmi' in col_name.lower():
        return gr.Number(label="BMI", value=25.0)
    elif 'gender' in col_name.lower():
        return gr.Radio(choices=["0 (Female / Other)", "1 (Male)"], label="Gender", value="0 (Female / Other)")
    elif 'residence' in col_name.lower():
        return gr.Radio(choices=["0 (Rural)", "1 (Urban)"], label="Residence Type", value="0 (Rural)")
    else:
        clean_label = col_name.replace('_', ' ').replace('Yes', '').title().strip()
        return gr.Radio(choices=["0 (No)", "1 (Yes)"], label=clean_label, value="0 (No)")

# 5. Custom CSS
liquid_glass_css = """
body, .gradio-container { background: radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%), linear-gradient(135deg, #1e0b36 0%, #0c1631 100%) !important; background-attachment: fixed !important; color: #ffffff !important; }
.glass-panel { background: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(16px) !important; -webkit-backdrop-filter: blur(16px) !important; border: 1px solid rgba(255, 255, 255, 0.15) !important; border-radius: 20px !important; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important; padding: 25px; }
.glass-panel h1, .glass-panel h2, .glass-panel h3, .glass-panel p, .glass-panel span, .glass-panel label { color: #ffffff !important; text-shadow: 0 1px 2px rgba(0,0,0,0.3); }
button.primary { background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.05) 100%) !important; backdrop-filter: blur(10px) !important; border: 1px solid rgba(255, 255, 255, 0.3) !important; border-radius: 15px !important; box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important; color: white !important; font-size: 16px !important; transition: all 0.3s ease-in-out !important; }
button.primary:hover { background: linear-gradient(135deg, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.15) 100%) !important; transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important; }
input[type="number"], input[type="text"], textarea { background: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; color: white !important; border-radius: 10px !important; }
"""

# 6. Build UI (Fixed for Gradio 6.0)
with gr.Blocks() as app:
    with gr.Column(elem_classes="glass-panel"):
        gr.Markdown("<h1 style='text-align: center;'>⚕️ AI Stroke Prediction Clinical Dashboard</h1>")
        gr.Markdown("<p style='text-align: center;'>Complete the patient intake form below. The system automatically engineers interaction features, applies standard scaling, and assesses stroke risk.</p>")
        
        ui_elements_dict = {}
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("👤 1. Patient Demographics", open=True):
                    for col in group_demographics: ui_elements_dict[col] = create_friendly_input(col)
                with gr.Accordion("🩺 2. Vitals & Medical History", open=True):
                    for col in group_vitals: ui_elements_dict[col] = create_friendly_input(col)
                with gr.Accordion("💼 3. Employment Status", open=True):
                    gr.Markdown("<div style='background: rgba(255, 165, 0, 0.2); padding: 8px; border-radius: 8px; border: 1px solid rgba(255, 165, 0, 0.4); margin-bottom: 10px;'><b>⚠️ Important Note:</b> Please select <b>'1 (Yes)' for ONLY ONE</b> of the employment types.</div>")
                    for col in group_work: ui_elements_dict[col] = create_friendly_input(col)
                with gr.Accordion("🚬 4. Lifestyle & Other", open=True):
                    for col in group_lifestyle + group_other: ui_elements_dict[col] = create_friendly_input(col)
                    
                submit_btn = gr.Button("🧠 Analyze Patient Risk", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 AI Assessment Result")
                output_display = gr.Textbox(label="Diagnosis & Recommendation", lines=6, text_align="center")
                gr.Markdown("""<br><div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);"><h3 style="margin-top: 0;">🤖 Model Architecture Details</h3><p><b>Primary Engine:</b> Soft-Voting Machine Learning Ensemble</p><ul><li>⚡ Logistic Regression</li><li>🌲 Random Forest Classifier</li><li>🚀 XGBoost Classifier</li></ul><p><b>Data Processing:</b> StandardScaler + SMOTE</p></div>""")
                
        final_input_list = [ui_elements_dict[col] for col in user_cols]
        submit_btn.click(fn=predict_stroke, inputs=final_input_list, outputs=output_display)

# 7. LAUNCH COMMAND FOR RENDER (Fixed for Gradio 6.0)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(server_name="0.0.0.0", server_port=port, css=liquid_glass_css, theme=gr.themes.Base(neutral_hue="slate"))
