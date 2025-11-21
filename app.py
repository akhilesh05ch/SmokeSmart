import re
import json
import math
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------- CONFIG ----------
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load ML Model and Scaler
print("Loading ML model...")
try:
    cvd_model = keras.models.load_model('cvd_model.h5')
    scaler = joblib.load('scaler.pkl')
    with open('model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    print("✓ ML model loaded successfully!")
    print(f"✓ Model accuracy: {model_metadata['test_accuracy']:.2%}")
    print(f"✓ Model AUC: {model_metadata['test_auc']:.4f}")
except Exception as e:
    print(f"✗ Error loading ML model: {e}")
    print("Please run train_model.py first to generate the model files.")
    cvd_model = None
    scaler = None
    model_metadata = None

SYSTEM_PROMPT = """You are a compassionate and caring cardiovascular health advisor.

**FIRST**, ask: "Would you like me to assess your cardiovascular disease risk, or discuss other health concerns?"

**IF USER WANTS RISK ASSESSMENT:**
Collect these 5 pieces of information (ask naturally, one at a time):
1. Age
2. Gender (male/female)
3. Current smoking status (yes/no)
4. Cigarettes per day (only if currently smoking)
5. Diabetes status (yes/no)

**CRITICAL RULES:**
- Ask questions warmly and naturally
- NEVER repeat questions you've already asked
- The system will automatically calculate and show the risk when all data is collected
- After thanking the user for providing information, DO NOT ask for any information again
- If the user asks "give me the score" or "calculate it", simply say "One moment, calculating your risk now..."

Talk to the user about whatever health concern he/she has. Engage in conversation after the risk score empathetically.

Be supportive and non-judgmental throughout."""

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
            "calculation_done": False,
            "asked_fields": set(),  # Track what we've asked for
            "last_question": None  # Track the last question asked
        }
    return chat_sessions[session_id]

# ------------------ Field Parsing ------------------

def parse_message_for_fields(fields, text, asked_fields, last_question):
    """Enhanced field parsing - returns True if new data was added"""
    t = text.lower()
    data_added = False
    
    # AGE
    if 'age' not in fields:
        age_patterns = [
            r'\b(\d{2})\s*(?:years?\s*old|yrs?|y/?o)\b',
            r'\bage\s*(?:is\s*)?(\d{2})\b',
            r'\bi\s*(?:am|\'m)\s*(\d{2})\b',
            r'(?:^|\s)(\d{2})(?:\s|$|\.)',
        ]
        
        for pattern in age_patterns:
            m = re.search(pattern, t)
            if m:
                try:
                    age = int(m.group(1))
                    if 18 <= age <= 120:
                        fields['age'] = age
                        asked_fields.add('age')
                        data_added = True
                        print(f"✓ CAPTURED AGE: {age}")
                        break
                except:
                    pass

    # GENDER
    if 'male' not in fields:
        if re.search(r'\b(?:i\s*(?:am|\'m)\s*(?:a\s*)?)?male\b', t) and not re.search(r'female', t):
            fields['gender'] = 'male'
            fields['male'] = 1
            asked_fields.add('gender')
            data_added = True
            print(f"✓ CAPTURED GENDER: male")
        elif re.search(r'\b(?:i\s*(?:am|\'m)\s*(?:a\s*)?)?female\b', t):
            fields['gender'] = 'female'
            fields['male'] = 0
            asked_fields.add('gender')
            data_added = True
            print(f"✓ CAPTURED GENDER: female")

    # SMOKING STATUS
    if 'currentSmoker' not in fields:
        # Check for YES (current smoker)
        if re.search(r'\b(?:yes|yeah|yep|i\s+do|i\s+am|currently|still)(?:\s+smok|\s*$)', t):
            fields['currentSmoker'] = 1
            fields['smoking_status'] = 'current'
            asked_fields.add('smoking')
            data_added = True
            print(f"✓ CAPTURED SMOKING: YES (current)")
        # Check for NO (not smoking)
        elif re.search(r'\b(?:no|nope|nah|don\'?t|do\s+not|never|quit|stopped|former)', t):
            fields['currentSmoker'] = 0
            fields['cigsPerDay'] = 0
            fields['smoking_status'] = 'never' if 'never' in t else 'former'
            asked_fields.add('smoking')
            asked_fields.add('cigarettes')  # Don't need to ask
            data_added = True
            print(f"✓ CAPTURED SMOKING: NO")

    # CIGARETTES PER DAY (only if current smoker)
    if fields.get('currentSmoker') == 1 and 'cigsPerDay' not in fields:
        cig_patterns = [
            r'(\d{1,2})\s*(?:cigarettes?|cigs?|per\s*day|a\s*day)',
            r'(?:smoke|smoking)\s*(?:about|around)?\s*(\d{1,2})',
            r'(?:^|\s)(\d{1,2})(?:\s|$)',
        ]
        
        for pattern in cig_patterns:
            m = re.search(pattern, t)
            if m:
                try:
                    cpd = int(m.group(1))
                    if 0 <= cpd <= 100:
                        fields['cigsPerDay'] = cpd
                        asked_fields.add('cigarettes')
                        data_added = True
                        print(f"✓ CAPTURED CIGARETTES: {cpd}")
                        break
                except:
                    pass

    # DIABETES - Handle both explicit mentions and yes/no when it was the last question
    if 'diabetes' not in fields:
        # If the last question was about diabetes and user gives yes/no
        if last_question == 'diabetes':
            if re.search(r'\b(?:yes|yeah|yep|yup|i\s+do|i\s+have|i\s+am)\b', t):
                fields['diabetes'] = 1
                asked_fields.add('diabetes')
                data_added = True
                print(f"✓ CAPTURED DIABETES: YES (from yes/no answer)")
            elif re.search(r'\b(?:no|nope|nah|nop|don\'?t|do\s+not|not)\b', t):
                fields['diabetes'] = 0
                asked_fields.add('diabetes')
                data_added = True
                print(f"✓ CAPTURED DIABETES: NO (from yes/no answer)")
        # Only match if "diabetes" or "diabetic" is explicitly mentioned
        elif re.search(r'diabet', t):
            if re.search(r'\b(?:yes|yeah|yep|i\s+have|i\s+am)\b', t):
                fields['diabetes'] = 1
                asked_fields.add('diabetes')
                data_added = True
                print(f"✓ CAPTURED DIABETES: YES")
            elif re.search(r'\b(?:no|nope|nah|don\'?t|do\s+not|not)\b', t):
                fields['diabetes'] = 0
                asked_fields.add('diabetes')
                data_added = True
                print(f"✓ CAPTURED DIABETES: NO")
        # Also check for standalone "diabetic" confirmation
        elif re.search(r'\bdiabetic\b', t):
            fields['diabetes'] = 1
            asked_fields.add('diabetes')
            data_added = True
            print(f"✓ CAPTURED DIABETES: YES (diabetic)")
        # Check for explicit "not diabetic" without the question
        elif re.search(r'not\s+diabetic', t):
            fields['diabetes'] = 0
            asked_fields.add('diabetes')
            data_added = True
            print(f"✓ CAPTURED DIABETES: NO (not diabetic)")

    return data_added

def check_mandatory_fields(fields):
    """Check if all mandatory fields are present"""
    required = ['age', 'male', 'currentSmoker', 'diabetes']
    
    for field in required:
        if field not in fields:
            return False
    
    # If current smoker, need cigarettes
    if fields.get('currentSmoker') == 1 and 'cigsPerDay' not in fields:
        return False
    
    # Ensure non-smokers have cigsPerDay = 0
    if fields.get('currentSmoker') == 0:
        fields['cigsPerDay'] = 0
    
    return True

# ------------------ ML-Based Risk Calculator ------------------

def calculate_ml_risk(fields):
    """Calculate CVD risk using trained neural network"""
    
    if cvd_model is None or scaler is None or model_metadata is None:
        return None, "Model Error", "ML model not loaded.", []
    
    if not check_mandatory_fields(fields):
        return None, "Insufficient Data", "Missing required information.", []
    
    all_features = model_metadata['all_features']
    feature_vector = []
    factors = []
    
    for feature in all_features:
        if feature in fields:
            feature_vector.append(fields[feature])
        else:
            mean_val = model_metadata['feature_means'].get(feature, 0)
            feature_vector.append(mean_val)
    
    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    risk_prob = cvd_model.predict(X_scaled, verbose=0)[0][0]
    risk_percentage = float(risk_prob * 100)
    
    # Build factors list
    factors.append(f"Age: {fields.get('age')} years")
    factors.append(f"Gender: {'Male' if fields.get('male') == 1 else 'Female'}")
    
    if fields.get('currentSmoker') == 1:
        factors.append(f"Current smoker: {fields.get('cigsPerDay')} cigarettes/day ⚠️ MAJOR RISK")
    else:
        factors.append("Non-smoker ✓")
    
    if fields.get('diabetes') == 1:
        factors.append("Diabetes: Yes ⚠️")
    else:
        factors.append("Diabetes: No ✓")
    
    # Risk level
    if risk_percentage < 10:
        level = "Low"
        explanation = "Your 10-year cardiovascular disease risk is LOW. Keep up the healthy habits!"
    elif risk_percentage < 20:
        level = "Moderate"
        explanation = "Your 10-year cardiovascular disease risk is MODERATE. Lifestyle changes can help reduce this."
    else:
        level = "High"
        explanation = "Your 10-year cardiovascular disease risk is HIGH. Please consult a healthcare provider."
    
    return risk_percentage, level, explanation, factors

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
        asked_fields = session['asked_fields']
        last_question = session.get('last_question')
        
        print(f"\n{'='*80}")
        print(f"📩 USER: {user_message}")
        print(f"📊 CURRENT FIELDS: {fields}")
        print(f"❓ ASKED FIELDS: {asked_fields}")
        print(f"🔍 LAST QUESTION: {last_question}")
        
        # Parse the message for new data
        data_added = parse_message_for_fields(fields, user_message, asked_fields, last_question)
        
        # Check if we now have all mandatory fields
        has_all = check_mandatory_fields(fields)
        
        print(f"✅ ALL DATA? {has_all}")
        print(f"🔢 CALCULATION DONE? {session.get('calculation_done')}")
        
        # IF WE HAVE ALL DATA AND HAVEN'T CALCULATED YET -> CALCULATE NOW!
        if has_all and not session.get('calculation_done'):
            print("🎯🎯🎯 TRIGGERING CALCULATION NOW! 🎯🎯🎯")
            session['calculation_done'] = True
            
            # Calculate the risk
            risk_score, risk_level, explanation, factors = calculate_ml_risk(fields)
            
            if risk_score is None:
                return jsonify({
                    'response': "I apologize, there was an error calculating your risk.",
                    'risk_score': None,
                    'risk_level': 'Error',
                    'risk_factors': [],
                    'session_id': session_id
                })
            
            print(f"💯 RISK CALCULATED: {risk_score:.1f}% ({risk_level})")
            
            # Build the complete response with results
            result_message = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **YOUR CARDIOVASCULAR RISK ASSESSMENT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**10-Year CVD Risk: {risk_score:.1f}%**
**Risk Level: {risk_level.upper()}**

{explanation}

**📊 Based on your information:**
{chr(10).join('  ▪ ' + f for f in factors)}

*Prediction accuracy: {model_metadata['test_accuracy']:.1%} (Framingham Heart Study dataset)*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{'🚨 **As a current smoker, quitting is the #1 way to reduce your risk! Would you like to discuss smoking cessation strategies?**' if fields.get('currentSmoker') == 1 else '💚 **Would you like to discuss ways to maintain or improve your heart health?**'}"""
            
            return jsonify({
                'response': result_message,
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'risk_factors': factors,
                'explanation': explanation,
                'session_id': session_id
            })
        
        # If already calculated, just continue conversation
        if session.get('calculation_done'):
            chat_obj = session['chat']
            response = chat_obj.send_message(user_message)
            
            return jsonify({
                'response': response.text,
                'risk_score': None,
                'risk_level': 'Completed',
                'risk_factors': [],
                'session_id': session_id
            })
        
        # Still collecting data - tell Gemini what we have and what we need
        needs = []
        next_question = None
        
        if 'age' not in fields:
            needs.append('age')
            next_question = 'age'
        elif 'male' not in fields:
            needs.append('gender')
            next_question = 'gender'
        elif 'currentSmoker' not in fields:
            needs.append('smoking status')
            next_question = 'smoking'
        elif fields.get('currentSmoker') == 1 and 'cigsPerDay' not in fields:
            needs.append('cigarettes per day')
            next_question = 'cigarettes'
        elif 'diabetes' not in fields:
            needs.append('diabetes')
            next_question = 'diabetes'
        
        # Store what question we're about to ask
        session['last_question'] = next_question
        
        context = f"\n\n[SYSTEM: You have: {list(asked_fields)}. Still need: {needs}. Ask for the FIRST missing item from the 'needs' list ONLY. Don't repeat questions.]"
        
        # Send to Gemini
        chat_obj = session['chat']
        response = chat_obj.send_message(user_message + context)
        bot_text = response.text
        
        # Show what we've collected
        collected = []
        if 'age' in fields:
            collected.append(f"✓ Age: {fields['age']}")
        if 'male' in fields:
            collected.append(f"✓ Gender: {fields['gender']}")
        if 'currentSmoker' in fields:
            collected.append(f"✓ Smoking: {'Yes' if fields['currentSmoker'] == 1 else 'No'}")
            if fields.get('currentSmoker') == 1 and 'cigsPerDay' in fields:
                collected.append(f"✓ Cigarettes: {fields['cigsPerDay']}/day")
        if 'diabetes' in fields:
            collected.append(f"✓ Diabetes: {'Yes' if fields['diabetes'] == 1 else 'No'}")
        
        return jsonify({
            'response': bot_text,
            'risk_score': None,
            'risk_level': f'Collecting data... ({len(collected)}/5)',
            'risk_factors': collected,
            'session_id': session_id
        })

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
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
