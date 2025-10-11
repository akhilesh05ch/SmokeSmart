import re
import json
import math
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------- CONFIG ----------
GEMINI_API_KEY = "AIzaSyD9qQPdaze2lPrAK0VnYBYLI3SGH5PSHhQ"
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a compassionate cardiovascular health advisor and smoking cessation counselor.

**FIRST**, always ask the user: "Would you like me to assess your cardiovascular disease risk, or would you prefer to discuss other health-related concerns?"

**IF USER WANTS RISK ASSESSMENT:**
- Ask ONE question at a time with warmth and empathy
- Collect these essential details naturally through conversation:
  * Age
  * Gender (male/female)
  * Smoking status (cigarettes per day, years smoking, or if they quit)
  * Blood pressure (if known)
  * Cholesterol levels (if known)
  * Family history of heart disease
  * Current symptoms (chest pain, shortness of breath, etc.)
  * Exercise habits
  
- When you have collected enough information (at minimum: age, gender, smoking history), say:
  "Thank you for sharing this information with me. Based on what you've told me, let me calculate your cardiovascular disease risk."
  
- Then output EXACTLY in this format:
  **CALCULATE_RISK**
  
- After that line, I (the backend) will compute the accurate risk and provide you the result. You will then explain it compassionately to the user.

**IF USER DOESN'T WANT RISK ASSESSMENT:**
- Act as a general medical health advisor
- Discuss their health concerns with empathy
- Provide evidence-based health information
- Encourage healthy lifestyle choices
- Support smoking cessation if relevant

Always be warm, non-judgmental, and supportive. Never attempt to calculate risk percentages yourself - always use the **CALCULATE_RISK** trigger when you have enough data."""

chat_sessions = {}

def start_model_session():
    model = genai.GenerativeModel('gemini-2.5-flash')
    chat = model.start_chat(history=[])
    chat.send_message(SYSTEM_PROMPT)
    return chat

def get_chat_session(session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "chat": start_model_session(), 
            "fields": {},
            "mode": None  # 'risk_assessment' or 'health_advisor'
        }
    return chat_sessions[session_id]

# ------------------ Field Parsing ------------------

def _num_from_range_or_single(s):
    if not s:
        return None
    s = s.strip()
    if '-' in s:
        parts = s.split('-')
        try:
            a = float(parts[0])
            b = float(parts[1])
            return int(round((a + b) / 2.0))
        except:
            return None
    try:
        return int(round(float(s)))
    except:
        return None

def parse_message_for_fields(fields, text):
    t = text.lower()

    # Gender
    if re.search(r"\b(male|man|boy|guy)\b", t) and not re.search(r"\bfemale\b", t):
        fields['gender'] = 'male'
    elif re.search(r"\b(female|woman|girl|lady)\b", t):
        fields['gender'] = 'female'

    # Age
    m = re.search(r"(?:age\s*(?:is)?\s*|i am\s+|i'm\s+|i'm\s+)?\b(\d{2,3})\b(?:\s*years\s*old|\s*yrs|\s*y/o)?", text, re.IGNORECASE)
    if m:
        age = int(m.group(1))
        if 18 <= age <= 120:
            fields['age'] = age

    # Years smoking
    m = re.search(r"(\d{1,3})\s*(?:years? smoking|yrs smoking|years? as a smoker|smoking for)", t)
    if m:
        yrs = _num_from_range_or_single(m.group(1))
        if yrs is not None:
            fields['years_smoking'] = yrs

    # Cigarettes per day
    m = re.search(r"(\d{1,3}(?:\s*-\s*\d{1,3})?)\s*(?:cigarettes|cigs|cig)\b", t)
    if not m:
        m = re.search(r"(?:smoke(?:s)?\s*(?:about|around)?\s*)(\d{1,3}(?:\s*-\s*\d{1,3})?)", t)
    if m:
        cpd = _num_from_range_or_single(m.group(1).replace(' ', ''))
        if cpd is not None:
            fields['cigarettes_per_day'] = cpd

    # Quit status
    if re.search(r"\b(quitted|quit|stopped smoking|i stopped|never smoked|don't smoke|dont smoke)\b", t):
        fields['cigarettes_per_day'] = 0
        fields['smoking_status'] = 'former' if 'quit' in t or 'stopped' in t else 'never'

    # Family history
    if re.search(r"\bfamily history\b|\bfather\b|\bmother\b|\bmom\b|\bdad\b|\bparents?\b", t):
        if re.search(r"\b(no family history|no history|family history: no)\b", t):
            fields['family_history'] = False
        else:
            fields['family_history'] = True

    # Symptoms
    if re.search(r"\bchest pain\b|\bchest tight\b|\bshortness of breath\b|\bsob\b|\bpalpitations\b", t):
        fields['symptoms'] = True

    # Blood pressure
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", t)
    if m:
        try:
            fields['bp_systolic'] = int(m.group(1))
            fields['bp_diastolic'] = int(m.group(2))
        except:
            pass

    # Cholesterol
    m = re.search(r"(?:cholesterol|total cholesterol)[^\d]{0,10}(\d{2,3})", t)
    if m:
        try:
            fields['cholesterol'] = int(m.group(1))
        except:
            pass

    # Exercise
    m = re.search(r"(\d{1,4})\s*(?:minutes|mins|min)\s*(?:per week|a week|weekly|daily|per day)?", t)
    if m and re.search(r"\b(exercise|walk|run|jog|gym|workout|cycle)\b", t):
        num = int(m.group(1))
        if re.search(r"per day|a day|daily", t):
            fields['exercise_min_per_week'] = num * 7
        else:
            fields['exercise_min_per_week'] = num

    if re.search(r"\b(no exercise|sedentary|never exercise)\b", t):
        fields['exercise_min_per_week'] = 0

    return fields

# ------------------ Framingham-Based Risk Calculator ------------------

def calculate_framingham_risk(fields):
    """
    Calculates 10-year CVD risk using simplified Framingham Risk Score
    Returns: (risk_percentage, risk_level, explanation, factors)
    """
    
    # Check if we have minimum required data
    required_fields = ['age', 'gender']
    if not all(k in fields for k in required_fields):
        return None, "Insufficient Data", "I need at least your age and gender to calculate risk.", []

    age = fields.get('age', 50)
    gender = fields.get('gender', 'male')
    
    # Initialize points (Framingham point system)
    points = 0
    factors = []
    
    # Age points (different for men and women)
    if gender == 'male':
        if age < 35: points += -1; age_pts = -1
        elif age < 40: points += 0; age_pts = 0
        elif age < 45: points += 1; age_pts = 1
        elif age < 50: points += 2; age_pts = 2
        elif age < 55: points += 3; age_pts = 3
        elif age < 60: points += 4; age_pts = 4
        elif age < 65: points += 5; age_pts = 5
        elif age < 70: points += 6; age_pts = 6
        else: points += 7; age_pts = 7
    else:  # female
        if age < 35: points += -9; age_pts = -9
        elif age < 40: points += -4; age_pts = -4
        elif age < 45: points += 0; age_pts = 0
        elif age < 50: points += 3; age_pts = 3
        elif age < 55: points += 6; age_pts = 6
        elif age < 60: points += 7; age_pts = 7
        elif age < 65: points += 8; age_pts = 8
        else: points += 8; age_pts = 8
    
    factors.append(f"Age: {age} years (baseline risk factor)")
    
    # Smoking points (most critical for your use case)
    cpd = fields.get('cigarettes_per_day', 0)
    years_smoking = fields.get('years_smoking', 0)
    smoking_status = fields.get('smoking_status', 'current')
    
    if cpd > 0 or years_smoking > 0:
        # Current smoker
        smoking_points = 4 if gender == 'male' else 3
        points += smoking_points
        factors.append(f"Current smoker: {cpd} cigarettes/day for {years_smoking} years (HIGH RISK)")
        
        # Additional points for heavy smoking
        if cpd >= 20:
            points += 2
            factors.append("Heavy smoking (20+ cigarettes/day) - additional risk")
    elif smoking_status == 'former':
        # Former smoker - reduced risk but still elevated
        points += 1
        factors.append("Former smoker - risk reducing over time")
    else:
        factors.append("Non-smoker - protective factor")
    
    # Blood pressure points
    systolic = fields.get('bp_systolic')
    if systolic:
        if systolic < 120:
            bp_pts = -2
        elif systolic < 130:
            bp_pts = 0
        elif systolic < 140:
            bp_pts = 1
        elif systolic < 160:
            bp_pts = 2
        else:
            bp_pts = 3
        points += bp_pts
        factors.append(f"Blood pressure: {systolic} mmHg")
    
    # Cholesterol points
    cholesterol = fields.get('cholesterol')
    if cholesterol:
        if cholesterol < 160:
            chol_pts = -3
        elif cholesterol < 200:
            chol_pts = 0
        elif cholesterol < 240:
            chol_pts = 1
        elif cholesterol < 280:
            chol_pts = 2
        else:
            chol_pts = 3
        points += chol_pts
        factors.append(f"Total cholesterol: {cholesterol} mg/dL")
    
    # Family history
    if fields.get('family_history'):
        points += 2
        factors.append("Family history of heart disease")
    
    # Symptoms present
    if fields.get('symptoms'):
        points += 3
        factors.append("Current cardiovascular symptoms (requires medical attention)")
    
    # Exercise (protective factor)
    exercise = fields.get('exercise_min_per_week')
    if exercise is not None:
        if exercise >= 150:
            points -= 2
            factors.append(f"Regular exercise: {exercise} min/week (protective)")
        elif exercise > 0:
            points -= 1
            factors.append(f"Some exercise: {exercise} min/week")
        else:
            points += 1
            factors.append("Sedentary lifestyle (risk factor)")
    
    # Convert points to risk percentage (Framingham algorithm)
    if gender == 'male':
        # Men's risk calculation
        if points <= -3: risk = 1
        elif points <= -2: risk = 2
        elif points <= -1: risk = 2
        elif points == 0: risk = 3
        elif points == 1: risk = 4
        elif points == 2: risk = 4
        elif points == 3: risk = 6
        elif points == 4: risk = 7
        elif points == 5: risk = 9
        elif points == 6: risk = 11
        elif points == 7: risk = 14
        elif points == 8: risk = 18
        elif points == 9: risk = 22
        elif points == 10: risk = 27
        elif points == 11: risk = 33
        elif points == 12: risk = 40
        elif points == 13: risk = 47
        else: risk = min(56 + (points - 14) * 5, 95)
    else:
        # Women's risk calculation
        if points <= -2: risk = 1
        elif points <= 0: risk = 2
        elif points == 1: risk = 2
        elif points == 2: risk = 3
        elif points == 3: risk = 3
        elif points == 4: risk = 4
        elif points == 5: risk = 5
        elif points == 6: risk = 6
        elif points == 7: risk = 7
        elif points == 8: risk = 8
        elif points == 9: risk = 9
        elif points == 10: risk = 11
        elif points == 11: risk = 13
        elif points == 12: risk = 15
        elif points == 13: risk = 17
        elif points == 14: risk = 20
        elif points == 15: risk = 24
        else: risk = min(27 + (points - 16) * 3, 95)
    
    # Determine risk level
    if risk < 10:
        level = "Low"
        explanation = "Your 10-year cardiovascular disease risk is low. Continue healthy habits!"
    elif risk < 20:
        level = "Moderate"
        explanation = "Your 10-year cardiovascular disease risk is moderate. Lifestyle changes can significantly reduce this."
    else:
        level = "High"
        explanation = "Your 10-year cardiovascular disease risk is high. Please consult a healthcare provider and consider immediate lifestyle changes."
    
    return risk, level, explanation, factors

# ------------------ Routes ------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index_page():
    return render_template('index.html')

@app.route('/about.html')
def about_page():
    return render_template('about.html')

@app.route('/chat-assistant.html')
def chat_page():
    return render_template('chat-assistant.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        payload = request.json or {}
        user_message = (payload.get('message') or "").strip()
        session_id = payload.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        session = get_chat_session(session_id)
        fields = session['fields']
        
        # Parse user message for health data
        parse_message_for_fields(fields, user_message)
        
        # Send to Gemini
        chat_obj = session['chat']
        model_response = chat_obj.send_message(user_message)
        bot_text = model_response.text or ""
        
        # Check if bot wants to calculate risk
        if "**CALCULATE_RISK**" in bot_text:
            # Calculate the actual risk
            risk_score, risk_level, explanation, factors = calculate_framingham_risk(fields)
            
            if risk_score is None:
                # Insufficient data
                final_response = "I need a bit more information to calculate your risk accurately. Could you tell me your age and gender?"
                return jsonify({
                    'response': final_response,
                    'risk_score': None,
                    'risk_level': 'Unknown',
                    'risk_factors': ['Insufficient data for calculation'],
                    'session_id': session_id
                })
            
            # Format the result message for the bot to deliver
            result_message = f"""Based on the information you've provided, your 10-year cardiovascular disease risk is approximately **{risk_score}%** ({risk_level} risk).

{explanation}

**Key risk factors identified:**
{chr(10).join('• ' + f for f in factors)}

{'**Important:** If you experience chest pain, severe shortness of breath, or other concerning symptoms, please seek immediate medical attention.' if fields.get('symptoms') else ''}

Would you like to discuss ways to reduce your risk? I can provide personalized recommendations for smoking cessation, lifestyle changes, and next steps."""
            
            # Send this back to Gemini to deliver naturally
            follow_up = chat_obj.send_message(f"Please deliver this risk assessment to the user in a warm, compassionate way: {result_message}")
            
            return jsonify({
                'response': follow_up.text,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': factors,
                'explanation': explanation,
                'session_id': session_id
            })
        
        # Normal conversation - check if we should compute risk anyway for display
        risk_score, risk_level, explanation, factors = calculate_framingham_risk(fields)
        
        return jsonify({
            'response': bot_text,
            'risk_score': risk_score,
            'risk_level': risk_level if risk_score is not None else 'Unknown',
            'risk_factors': factors if risk_score is not None else ['Continue chatting to gather information'],
            'explanation': explanation if risk_score is not None else '',
            'session_id': session_id
        })

    except Exception as e:
        print("ERROR in /chat:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return jsonify({'message': 'Chat reset successfully'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)